import z from "zod";
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

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

const add = tool(({ a, b }): number => a + b, {
  name: "add",
  description: "Add numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const sqrt = tool(({ value }): number => Math.sqrt(value), {
  name: "sqrt",
  description: "Extract the square root of a number",
  schema: z.object({
    value: z.number().describe("The number to extract the square root from"),
  }),
});

const tools = [add, sqrt];

const modelWithTools = model.bindTools(tools);

const MessagesState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (left, right) => left.concat(right),
  }),
});

async function llmNode(state: typeof MessagesState.State) {
  const response = await modelWithTools.invoke([
    new SystemMessage(
      "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    ),
    ...state.messages,
  ]);

  return {
    messages: [response],
  };
}

const toolNode = new ToolNode(tools);

async function shouldContinue(state: typeof MessagesState.State) {
  const lastMessage = state.messages.at(-1);

  if (lastMessage == null || !isAIMessage(lastMessage)) return END;

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  return END;
}

const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmNode)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile();

const result = await agent.invoke({
  messages: [new HumanMessage("Calc sqrt(9 + 7)")],
});

console.log("~~~ result", JSON.stringify(result, null, 2));
