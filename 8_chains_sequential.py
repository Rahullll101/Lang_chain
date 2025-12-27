from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
# -------------------------------
# LLM Configuration
# -------------------------------
llm = model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash',temperature=0.0)


parser = StrOutputParser()

# -------------------------------
# Step 1: Explanation Prompt
# -------------------------------
explain_prompt = ChatPromptTemplate.from_template(
    """
    Explain the following topic in detail with examples:
    {topic}
    """
)

# -------------------------------
# Step 2: Summarization Prompt
# -------------------------------
summary_prompt = ChatPromptTemplate.from_template(
    """
    Summarize the explanation below into 3 concise bullet points:
    {explanation}
    """
)

# -------------------------------
# Sequential Chain
# Output of step 1 becomes input of step 2
# -------------------------------
sequential_chain = (
    explain_prompt
    | llm
    | parser
    | (lambda explanation: {"explanation": explanation})
    | summary_prompt
    | llm
    | parser
)

# -------------------------------
# Execution with Timing
# -------------------------------
def run_sequential_chain(topic: str):
    result = sequential_chain.invoke({
        "topic": topic
    })

    print("\n📌 Final Output:")
    print(result)

# -------------------------------
# Run Example
# -------------------------------
topic = "The impact of artificial intelligence on modern healthcare"
run_sequential_chain(topic)


# User Topic
#    ↓
# Explain Prompt
#    ↓
# LLM
#    ↓
# Explanation Text
#    ↓
# Summary Prompt
#    ↓
# LLM
#    ↓
# Final Summary
