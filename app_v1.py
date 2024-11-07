import streamlit as st
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
import os
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from llama_index.llms.openllm import OpenLLM

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize models
def init_models():
    gpt_4_mini = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini-2024-07-18",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0.1
    )


    O1_mini=AzureChatOpenAI(
        azure_deployment="o1-mini-2024-09-12",
        api_key=os.getenv("AZURE_OPENAI_API_KEY_O1"),
        api_version=os.getenv("AZURE_API_VERSION_O1"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_O1"),
        temperature=1
    )

    # mixtral_model = ChatGroq(
    #     groq_api_key=os.getenv("GROQ_API_KEY"),
    #     model_name="mixtral-8x7b-32768",streaming = True
    # )

    mixtral_model = OpenLLM(
            api_base="http://mistral7b.cloudverse.freshworkscorp.com/v1/",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            api_key="na",
            max_tokens=4096,
            temperature=0.1
          )
    
    gemma_model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="gemma2-9b-it",streaming = True
    )
    # llama_model = ChatGroq(
    #     groq_api_key=os.getenv("GROQ_API_KEY"),
    #     model_name="llama-3.2-90b-text-preview",streaming = True
    # )
    
    llama_model = OpenLLM(
            api_base="http://llama70bq.cloudverse.freshworkscorp.com/v1",
            model="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16",  #meta-llama/Meta-Llama-3-8B-Instruct
            api_key="na",
            max_tokens=4096,
            temperature=0.1
          )
    return gpt_4_mini, mixtral_model, llama_model , O1_mini, gemma_model

# Setup workflow
class GraphState(TypedDict):
    question: Optional[str] = None
    classification: Optional[str] = None
    response: Optional[str] = None
    model: Optional[str] = None

def create_workflow(gpt_4_mini, mixtral_model,llama_model,O1_mini,gemma_model):
    workflow = StateGraph(GraphState)
    
    # def analyze_complexity(question):
    #     response = gpt_4_mini.invoke(
    #         f"Classify the complexity of this question as 'simple' or 'complex' or 'reasoning'. Just output the classification. Question: {question}"
    #     )
    #     return {"classification": response.content.strip()}
    
    def analyze_complexity(question):                
        prompt = f"""                
                Classify the complexity of this question as 'simple', 'complex', 'reasoning', or 'coding'. Just output the classification.                
                                
                Here are some examples:                
                                
                Simple:                
                Q: What is the capital of Japan?                
                A: simple                
                                
                Q: How many continents are there on Earth?                
                A: simple                
                                
                Q: Who wrote the play "Romeo and Juliet"?                
                A: simple                
                                
                Q: What is the chemical symbol for gold?                
                A: simple                
                                
                Q: In which year did World War II end?                
                A: simple                
                                
                Complex:                
                Q: Explain the theory of relativity.                
                A: complex                
                                
                Q: How does the human immune system work?                
                A: complex                
                                
                Q: What are the main causes and effects of climate change?                
                A: complex                
                                
                Q: Describe the process of protein synthesis in cells.                
                A: complex                
                                
                Q: What are the key differences between various economic systems like capitalism, socialism, and communism?                
                A: complex                
                                
                Reasoning:                
                Q: If all mammals are warm-blooded, and all whales are mammals, what can we conclude about whales?                
                A: reasoning                
                                
                Q: A train travels 120 km in 2 hours. What is its average speed in km/h?                
                A: reasoning                
                                
                Q: If a red sock turns all white clothes pink in the wash, and you have 3 white shirts and 2 red socks, how many pink shirts will you end up with?                
                A: reasoning                
                                
                Q: In a room of 30 people, what's the probability that at least two people share the same birthday?                
                A: reasoning                
                                
                Q: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?                
                A: reasoning                
                                
                Coding:                
                Q: Write a function to check if a string is a palindrome.                
                A: coding                
                                
                Q: Implement a binary search algorithm.                
                A: coding                
                                
                Q: Create a class representing a basic calculator with add, subtract, multiply, and divide methods.                
                A: coding                
                                
                Q: Write a script to find the most frequent word in a text file.                
                A: coding                
                                
                Q: Implement a simple linked list data structure with methods for insertion and deletion.                
                A: coding                
                                
                Now, classify the following question:                
                Question: {question}                
                """                
        response = gpt_4_mini.invoke(prompt)                
        return {"classification": response.content.strip()}
    

    def handle_simple_query(state):
        question = state['question']
        response = gemma_model.invoke(question)
        return {"response": response.content, "model": "gemma2-9b-it"}

    def handle_complex_query(state):
        question = state['question']
        response = llama_model.complete(question)
        return {"response": response.text, "model": "Llama-3.1-70B-Instruct-quantized"}

    def handle_reasoning_query(state):
        question = state['question']
        # prompt = f"""Please approach this question step by step:
        # 1. First, identify the key components of the problem  
        # 2. Then, break down the logical steps needed
        # 3. Finally, provide a well-reasoned conclusion

        # Question: {question}
        # """
        response = O1_mini.invoke(question)
        return {"response": response.content, "model": "o1-mini"}
    
    def handle_coding_query(state):
        question = state['question']
        response = mixtral_model.complete(question)
        return {"response": response.text, "model": "Mistral-7B-Instruct-v0.2"}
    
    

    workflow.add_node("analyze_complexity", analyze_complexity)
    workflow.add_node("handle_simple", handle_simple_query)
    workflow.add_node("handle_complex", handle_complex_query)
    workflow.add_node("handle_reasoning", handle_reasoning_query)
    workflow.add_node("handle_coding", handle_coding_query)

    

    workflow.set_entry_point("analyze_complexity")

    workflow.add_edge("handle_simple", END)
    workflow.add_edge("handle_complex", END)
    workflow.add_edge("handle_reasoning", END)
    workflow.add_edge("handle_coding", END)

    def route_to_model(state):
        if state["classification"] == "simple":
            return "handle_simple"
        elif state["classification"] == "reasoning":
            return "handle_reasoning"
        elif state["classification"] == "coding":
            return "handle_coding"
        else:
            return "handle_complex"

    workflow.add_conditional_edges(
        "analyze_complexity",
        route_to_model,
        {
            "handle_simple": "handle_simple",
            "handle_complex": "handle_complex",
            "handle_reasoning": "handle_reasoning",
            "handle_coding":"handle_coding"
        }
    )
    
    return workflow.compile()

# Streamlit app
def main():
    st.set_page_config(page_title="Route LLM", page_icon="🤖") #layout="wide"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "models" not in st.session_state:
        st.session_state.models = init_models()
    if "workflow" not in st.session_state:
        st.session_state.workflow = create_workflow(*st.session_state.models)

    st.title("🤖 LLM Routing")
    st.markdown("""
    This app automatically routes your questions to the most appropriate LLM based on complexity:
    - Simple queries → Gemma
    - Complex queries → Llama
    - Reasoning queries → o1-mini
    - Coding queries → Mistral
    """)

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "model" in message:
                st.caption(f"Responded using: {message['model']}")

    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from workflow
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.workflow.invoke({
                    "question": prompt
                })
                print(response)
                
                st.write(response["response"])
             
                st.caption(f"Responded using: {response['model']}")
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"],
                    "model": response["model"]
                })

    # # Sidebar with chat history stats
    # with st.sidebar:
    #     st.header("Chat Statistics")
    #     total_messages = len(st.session_state.messages)
    #     st.metric("Total Messages", total_messages)
        
    #     if total_messages > 0:
    #         model_usage = {}
    #         for message in st.session_state.messages:
    #             if "model" in message:
    #                 model_usage[message["model"]] = model_usage.get(message["model"], 0) + 1
            
    #         st.subheader("Model Usage")
    #         for model, count in model_usage.items():
    #             st.text(f"{model}: {count} times")

    #     if st.button("Clear Chat History"):
    #         st.session_state.messages = []
    #         st.rerun()

if __name__ == "__main__":
    main()
