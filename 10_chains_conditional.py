from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# LLM Configuration
# -------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

parser = StrOutputParser()

# -------------------------------------------------
# 1. LLM Classifier
# -------------------------------------------------
classification_prompt = ChatPromptTemplate.from_template(
    """
    You are an intent classifier.

    Classify the user input into exactly ONE of these categories:

    - technical : APIs, backend, system design, architecture, coding
    - general   : normal questions needing explanation
    - greeting  : greetings, small talk, casual conversation

    User input:
    {question}

    Respond with ONLY one word:
    technical | general | greeting
    """
)

classifier_chain = classification_prompt | llm | parser

# -------------------------------------------------
# 2. Destination Chains
# -------------------------------------------------
technical_chain = (
    ChatPromptTemplate.from_template(
        """
        Explain the following topic in a technical manner
        with examples and best practices:

        {question}
        """
    )
    | llm
    | parser
)

general_chain = (
    ChatPromptTemplate.from_template(
        """
        Explain the following in simple, beginner-friendly terms:

        {question}
        """
    )
    | llm
    | parser
)

greeting_chain = (
    ChatPromptTemplate.from_template(
        """
        Reply politely and conversationally to the user:

        {question}
        """
    )
    | llm
    | parser
)

# -------------------------------------------------
# 3. SAFE ROUTER
# -------------------------------------------------
def route_with_context(input):
    intent = classifier_chain.invoke({"question": input["question"]}).strip().lower()

    return {
        "intent": intent,
        "question": input["question"]
    }

# -------------------------------------------------
# 4. Conditional Chain (CORRECT)
# -------------------------------------------------
conditional_chain = (
    RunnableLambda(route_with_context)
    | RunnableLambda(
        lambda x:
        technical_chain.invoke(x)
        if x["intent"] == "technical"
        else general_chain.invoke(x)
        if x["intent"] == "general"
        else greeting_chain.invoke(x)
    )
)

# -------------------------------------------------
# 5. Execution with Timing
# -------------------------------------------------
def run_conditional_chain(question: str):
    print("\n==============================")
    print("User Question:", question)

    result = conditional_chain.invoke({
        "question": question
    })

    print("\n Final Output:")
    print(result)


# -------------------------------------------------
# 6. Run Examples
# -------------------------------------------------
# Technical example
run_conditional_chain(
    "How does API rate limiting work in distributed systems?"
)

# General example
run_conditional_chain(
    "How are you ?"
)

# User Question
#      ↓
# LLM Intent Classifier
# (technical | general | greeting)
#      ↓
# Context Preserved (intent + original question)
#      ↓
#  ┌──────────────┬──────────────┬──────────────┐
#  │  Technical   │   General    │   Greeting   │
#  │  Chain       │   Chain      │   Chain      │
#  └──────────────┴──────────────┴──────────────┘

