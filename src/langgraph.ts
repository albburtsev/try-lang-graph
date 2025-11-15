import z from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { registry } from "@langchain/langgraph/zod";
import { tool } from "@langchain/core/tools";
import { Annotation, StateGraph, END, START } from "@langchain/langgraph";
import {
  isAIMessage,
  BaseMessage,
  SystemMessage,
  ToolMessage,
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

const toolsByName = {
  [add.name]: add,
};

const modelWithTools = model.bindTools(Object.values(toolsByName));

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

async function toolNode(state: typeof MessagesState.State) {
  const lastMessage = state.messages.at(-1);

  if (lastMessage == null || !isAIMessage(lastMessage)) {
    return { messages: [] };
  }

  const result: ToolMessage[] = [];

  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name];
    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  return { messages: result };
}

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
  messages: [new HumanMessage("Add 3 and 4.")],
});

console.log('~~~ result', JSON.stringify(result, null, 2))
