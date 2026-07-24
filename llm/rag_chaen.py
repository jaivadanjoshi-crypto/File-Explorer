from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import config

SYSTEM_PROMPT = """
You are an intelligent AI assistant that answers questions using only the provided context.

Instructions:
- Use the retrieved context as your primary source of information.
- Provide clear, accurate, and well-structured answers.
- If the answer is not present in the context, respond with:
  "I couldn't find that information in the provided document."
- Do not make up facts or rely on outside knowledge.
- If the context contains only partial information, clearly state what is available and mention any missing details.
- When appropriate, present information using bullet points or numbered lists for better readability.

Retrieved Context:
{context}
"""

def build_rag_chain(vectorstore: FAISS):
    """Builds and returns the QA retrieval chain."""
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=0,
        groq_api_key=config.GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, combine_docs_chain)