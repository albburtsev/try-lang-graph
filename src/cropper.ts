import z from "zod";
import fs from "fs";
import path from "path";
import sharp from "sharp";
import { fileURLToPath } from "url";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, StateGraph, END, START } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  isAIMessage,
  BaseMessage,
  SystemMessage,
  HumanMessage,
} from "@langchain/core/messages";

const getImageFormat = async (buffer: Buffer): Promise<string> => {
  const metadata = await sharp(buffer).metadata();
  return metadata.format;
};

const getImageData = async (filename: string): Promise<string> => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const filepath = path.resolve(__dirname, filename);

  const buffer = fs.readFileSync(filepath);
  const format = await getImageFormat(buffer);

  return `data:image/${format};base64,${buffer.toString("base64")}`;
};

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

const MessagesState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (left, right) => left.concat(right),
  }),
  sourceImageDataUri: Annotation<string | null>({
    reducer: (_, right) => right ?? null,
    default: () => null,
  }),
  croppedImageDataUri: Annotation<string | null>({
    reducer: (_, right) => right ?? null,
    default: () => null,
  }),
  approval: Annotation<{
    status: "pending" | "approved" | "rejected";
    feedback: string | null;
  }>({
    reducer: (_, right) => right ?? { status: "pending", feedback: null },
    default: () => ({ status: "pending", feedback: null }),
  }),
});

const cropImage = tool(
  async ({ cropArea }, config) => {
    // Access state from config.configurable
    const state = config?.configurable as
      | typeof MessagesState.State
      | undefined;
    const sourceImageDataUri = state?.sourceImageDataUri;

    if (!sourceImageDataUri) {
      throw new Error("No source image available to crop");
    }

    // Extract base64 data from data URI
    const base64Data = sourceImageDataUri.split(",")[1];
    const buffer = Buffer.from(base64Data, "base64");
    const format = await getImageFormat(buffer);

    // Crop the image using sharp
    const croppedBuffer = await sharp(buffer)
      .extract({
        left: cropArea.x,
        top: cropArea.y,
        width: cropArea.width,
        height: cropArea.height,
      })
      .toBuffer();

    // Return cropped image as data URI
    const croppedDataUri = `data:image/${format};base64,${croppedBuffer.toString(
      "base64"
    )}`;
    return croppedDataUri;
  },
  {
    name: "cropImage",
    description:
      "Crop the current image using the specified coordinates and dimensions. The tool will use the image from the conversation context.",
    schema: z.object({
      cropArea: z
        .object({
          x: z
            .number()
            .int()
            .min(0)
            .describe("X coordinate of the crop area (left)"),
          y: z
            .number()
            .int()
            .min(0)
            .describe("Y coordinate of the crop area (top)"),
          width: z
            .number()
            .int()
            .positive()
            .describe("Width of the crop area in pixels"),
          height: z
            .number()
            .int()
            .positive()
            .describe("Height of the crop area in pixels"),
        })
        .describe("Crop area coordinates and dimensions"),
    }),
  }
);

const tools = [cropImage];

const modelWithTools = model.bindTools(tools);

async function llmNode(state: typeof MessagesState.State) {

  // Extract the source image from messages
  let newSourceImageDataUri = state.sourceImageDataUri;

  for (const msg of state.messages) {
    // Extract source image
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === "image_url") {
          const imageUrl = (part as any).image_url?.url;
          if (imageUrl && typeof imageUrl === "string") {
            // Store the largest image as source (usually the original)
            if (
              !newSourceImageDataUri ||
              imageUrl.length > newSourceImageDataUri.length
            ) {
              newSourceImageDataUri = imageUrl;
            }
          }
        }
      }
    }
  }

  // System prompt for the LLM - independent and self-contained
  const systemPrompt =
    "You are a helpful assistant that can analyze images and crop them based on user requests. " +
    "Follow this workflow:\n" +
    "1. Analyze the original image to understand its content and dimensions\n" +
    "2. Determine the appropriate crop area (x, y, width, height) based on the user's requirements\n" +
    "3. Use the cropImage tool to perform the crop - you only need to provide the cropArea coordinates" +
    (state.approval.feedback
      ? `\n\nIMPORTANT: Previous crop attempt was rejected with this feedback:\n"${state.approval.feedback}"\nPlease adjust your crop parameters accordingly.`
      : "");

  const response = await modelWithTools.invoke([
    new SystemMessage(systemPrompt),
    ...state.messages,
  ]);

  return {
    messages: [response],
    sourceImageDataUri: newSourceImageDataUri,
    // Reset approval when starting new crop attempt
    approval: { status: "pending" as const, feedback: null },
  };
}

// Define the approval schema
const ApprovalSchema = z.object({
  approved: z.boolean().describe("Whether the crop is approved"),
  feedback: z.string().describe("Detailed explanation of the approval decision"),
});

// Approver node that independently validates the crop result
async function approverNode(state: typeof MessagesState.State) {
  console.log("\n=== Approver Node ===");

  if (!state.croppedImageDataUri || !state.sourceImageDataUri) {
    return {
      approval: {
        status: "rejected" as const,
        feedback: "No cropped image was produced",
      },
    };
  }

  // Independent approval prompt - evaluates crop quality on its own criteria
  const approvalPrompt = `You are an image crop quality validator. Your job is to compare the original image with the cropped result and determine if the crop is appropriate and high quality.

Evaluate the crop based on:
- Is the subject/main content properly framed and not cut off?
- Does the crop maintain good composition?
- Is the aspect ratio reasonable and intentional?
- Are important elements preserved in the crop?

If approved is true, provide a brief confirmation in feedback.
If approved is false, provide specific feedback about what's wrong (e.g., "The product is partially cut off on the right side" or "The crop removes important context from the top of the image").`;

  // Use structured output with withStructuredOutput
  const modelWithStructuredOutput = model.withStructuredOutput(ApprovalSchema, {
    name: "approval_result",
  });

  const response = await modelWithStructuredOutput.invoke([
    new HumanMessage({
      content: [
        {
          type: "text",
          text: approvalPrompt,
        },
        {
          type: "text",
          text: "Original image:",
        },
        {
          type: "image_url",
          image_url: {
            url: state.sourceImageDataUri,
          },
        },
        {
          type: "text",
          text: "Cropped image:",
        },
        {
          type: "image_url",
          image_url: {
            url: state.croppedImageDataUri,
          },
        },
      ],
    }),
  ]);

  console.log("Approver response:", response);

  return {
    approval: {
      status: response.approved ? ("approved" as const) : ("rejected" as const),
      feedback: response.feedback,
    },
  };
}

async function toolNode(state: typeof MessagesState.State) {
  // Create a ToolNode with state passed in config
  const toolNodeWithState = new ToolNode(tools);

  // Invoke with state in configurable
  const toolResult = await toolNodeWithState.invoke(state, {
    configurable: state,
  });

  // Extract cropped image from tool results
  let newCroppedImageDataUri = state.croppedImageDataUri;

  // Check the tool result messages for image data
  for (const msg of toolResult.messages) {
    const content = msg.content;

    if (typeof content === "string" && content.startsWith("data:image/")) {
      // Store the cropped image in state
      newCroppedImageDataUri = content;
      break;
    }
  }

  return {
    messages: toolResult.messages,
    croppedImageDataUri: newCroppedImageDataUri,
  };
}

// Routing after LLM node
async function routeAfterLLM(state: typeof MessagesState.State) {
  const lastMessage = state.messages.at(-1);

  if (lastMessage == null || !isAIMessage(lastMessage)) return END;

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  // If no tool calls, go to approver to validate the response
  return "approver";
}

// Node to add rejection feedback to messages before retrying
async function addRejectionFeedback(state: typeof MessagesState.State) {
  console.log("Adding rejection feedback to messages");
  return {
    messages: [
      new HumanMessage({
        content: `The crop was rejected. Feedback: ${state.approval.feedback}\n\nPlease try again with adjusted crop parameters.`,
      }),
    ],
  };
}

// Routing after approver node
async function routeAfterApprover(state: typeof MessagesState.State) {
  console.log(`\n=== Approval Status: ${state.approval.status} ===`);

  if (state.approval.status === "approved") {
    console.log("Routing to END (approved)");
    return END;
  }

  if (state.approval.status === "rejected") {
    console.log(`Rejection feedback: ${state.approval.feedback}`);
    console.log("Routing to addRejectionFeedback");
    return "addRejectionFeedback";
  }

  console.log("Routing to END (default)");
  return END;
}

const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmNode)
  .addNode("toolNode", toolNode)
  .addNode("approver", approverNode)
  .addNode("addRejectionFeedback", addRejectionFeedback)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", routeAfterLLM)
  .addEdge("toolNode", "approver")
  .addConditionalEdges("approver", routeAfterApprover)
  .addEdge("addRejectionFeedback", "llmCall")
  .compile();

const result = await agent.invoke({
  messages: [
    new HumanMessage({
      content: [
        {
          type: "text",
          text: "This is a product image. Crop it to a 9:16 ratio so that the product is not cropped. ",
        },
        {
          type: "image_url",
          image_url: {
            url: await getImageData("../assets/kettle.webp"),
          },
        },
      ],
    }),
  ],
});

console.log("\n=== Final Result ===");
console.log(`Approval Status: ${result.approval.status}`);
console.log(
  `Source image: ${result.sourceImageDataUri ? "Present" : "Not found"}`
);
console.log(
  `Cropped image: ${result.croppedImageDataUri ? "Present" : "Not found"}`
);
console.log(`Total messages: ${result.messages.length}`);

if (result.approval.feedback) {
  console.log(`\nApproval feedback: ${result.approval.feedback}`);
}

// Save cropped image if available and approved
if (result.croppedImageDataUri && result.approval.status === "approved") {
  const base64Data = result.croppedImageDataUri.split(",")[1];
  const outputPath = path.resolve(process.cwd(), "cropped-output.png");
  fs.writeFileSync(outputPath, Buffer.from(base64Data, "base64"));
  console.log(`\n✓ Cropped image saved to: ${outputPath}`);
} else if (result.approval.status === "rejected") {
  console.log(`\n✗ Crop was rejected. No output saved.`);
}
