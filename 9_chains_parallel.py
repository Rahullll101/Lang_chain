from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
# -------------------------------
# LLM Configuration
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

parser = StrOutputParser()

# -------------------------------
# Individual Parallel Tasks
# -------------------------------

# Task 1: Summary
summary_chain = (
    ChatPromptTemplate.from_template(
        "Summarize the following text:\n{text}"
    )
    | llm
    | parser
)

# Task 2: Keywords
keywords_chain = (
    ChatPromptTemplate.from_template(
        "Extract 5 important keywords:\n{text}"
    )
    | llm
    | parser
)

# Task 3: Sentiment Analysis
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Analyze sentiment (Positive / Negative / Neutral):\n{text}"
    )
    | llm
    | parser
)

# -------------------------------
# Parallel Chain
# -------------------------------
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain
)

# -------------------------------
# Execution with Timing
# -------------------------------
def run_parallel_chain(text: str):
    result = parallel_chain.invoke({
        "text": text
    })

    print("\n Parallel Outputs:")
    for k, v in result.items():
        print(f"\n{k.upper()}:\n{v}")


# -------------------------------
# Run Example
# -------------------------------
run_parallel_chain(
        "LangChain enables developers to build scalable applications powered by large language models."
    )

#             ┌─ Summary ───────┐
# Input Text ─┼─ Keywords ──────┼─► Combined Output
#             └─ Sentiment ─────┘
