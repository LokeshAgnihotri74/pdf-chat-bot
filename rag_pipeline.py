"""
Advanced 'Chat with your PDF' RAG Pipeline.
Features:
1. Robust PDF extraction with fallback for corrupted, encrypted, or empty/scanned PDFs.
2. Custom recursive document chunking with metadata.
3. ChromaDB vector store supporting Cosine Similarity and Euclidean Distance.
4. Strict Hallucination Guardrails ensuring zero hallucination.
5. Real-time streaming answers via Groq Cloud LLM.
"""

import io
import re
from typing import List, Dict, Any, Generator, Tuple, Optional
from dataclasses import dataclass
import pypdf
from pypdf.errors import PdfReadError, FileNotDecryptedError
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    GROQ_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    TOP_K_RESULTS,
    STRICT_GUARDRAIL_FALLBACK,
    DISTANCE_METRICS
)


# Custom Exceptions
class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed due to corruption, encryption, or format errors."""
    pass


class PDFEmptyError(PDFProcessingError):
    """Raised when a PDF contains no extractable text (e.g. image-only or scanned)."""
    pass


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]


def extract_text_from_pdf(file_source: Any, filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text page by page from a PDF file path or file-like stream (BytesIO).
    Explicitly handles corrupted, password-protected, or empty PDFs.

    Returns:
        List of dicts: [{'page_num': int, 'text': str, 'filename': str}]
    """
    if file_source is None:
        raise PDFProcessingError("No PDF file was provided.")

    try:
        # Wrap bytes or BytesIO if needed
        if isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        elif hasattr(file_source, "read"):
            stream = file_source
        elif isinstance(file_source, str):
            stream = open(file_source, "rb")
        else:
            raise PDFProcessingError(f"Unsupported file source type: {type(file_source)}")

        reader = pypdf.PdfReader(stream)

        # Check for encryption / password lock
        if reader.is_encrypted:
            try:
                # Try decrypting with empty password
                reader.decrypt("")
            except Exception:
                raise PDFProcessingError(
                    f"The PDF '{filename}' is password-protected or encrypted. Please unlock it before uploading."
                )

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise PDFEmptyError(f"The PDF '{filename}' contains 0 pages.")

        extracted_pages = []
        total_extracted_chars = 0

        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as page_err:
                page_text = f"[Warning: Text extraction failed on page {idx + 1}: {str(page_err)}]"

            clean_text = page_text.strip()
            total_extracted_chars += len(clean_text)
            extracted_pages.append({
                "page_num": idx + 1,
                "text": clean_text,
                "filename": filename
            })

        # Explicit fallback for scanned or empty PDFs
        if total_extracted_chars == 0:
            raise PDFEmptyError(
                f"The PDF '{filename}' has {num_pages} page(s) but no readable text could be extracted. "
                "It may be a scanned document, image-only file, or contains non-standard font encodings."
            )

        return extracted_pages

    except (PdfReadError, FileNotDecryptedError) as e:
        raise PDFProcessingError(f"Corrupted or invalid PDF file '{filename}': {str(e)}")
    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(f"Unexpected error while reading PDF '{filename}': {str(e)}")


def recursive_split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Custom recursive text chunker that preserves semantic boundaries
    using cascading split characters (\n\n, \n, sentence end, space).
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the highest-priority separator present in text
    separator = ""
    for sep in separators:
        if sep == "":
            separator = ""
            break
        if sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator != "" else list(text)

    chunks = []
    current_chunk = []
    current_len = 0

    for piece in splits:
        piece_len = len(piece) + (len(separator) if current_chunk else 0)
        if current_len + piece_len > chunk_size and current_chunk:
            combined = separator.join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Implement overlap by keeping tail elements
            overlap_acc = []
            overlap_len = 0
            for prev in reversed(current_chunk):
                if overlap_len + len(prev) <= chunk_overlap:
                    overlap_acc.insert(0, prev)
                    overlap_len += len(prev) + len(separator)
                else:
                    break
            current_chunk = list(overlap_acc)
            current_len = sum(len(p) for p in current_chunk) + (len(separator) * max(0, len(current_chunk) - 1))

        current_chunk.append(piece)
        current_len += piece_len

    if current_chunk:
        combined = separator.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[DocumentChunk]:
    """
    Transforms extracted pages into chunked DocumentChunk objects with rich metadata.
    """
    all_chunks = []
    global_chunk_idx = 1

    for page in pages:
        page_num = page["page_num"]
        filename = page["filename"]
        text = page["text"]

        if not text:
            continue

        raw_chunks = recursive_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, raw_chunk in enumerate(raw_chunks, start=1):
            chunk = DocumentChunk(
                text=raw_chunk,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "chunk_id": f"{filename}_p{page_num}_c{chunk_idx}",
                    "global_idx": global_chunk_idx,
                    "char_count": len(raw_chunk)
                }
            )
            all_chunks.append(chunk)
            global_chunk_idx += 1

    return all_chunks


class ChromaRAGManager:
    """
    Manages vector indexing and retrieval with ChromaDB.
    Supports dynamic distance metrics:
      - 'cosine': Cosine Distance (1 - Cosine Similarity)
      - 'l2': Squared Euclidean Distance
    Uses lightweight local default embeddings (all-MiniLM-L6-v2 via ONNX).
    """

    def __init__(self, collection_name: str = "pdf_rag_collection", distance_metric: str = "cosine"):
        if distance_metric not in ["cosine", "l2"]:
            raise ValueError(f"Unsupported distance metric: {distance_metric}. Must be 'cosine' or 'l2'.")

        self.distance_metric = distance_metric
        self.collection_name = collection_name
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Initialize or recreate collection with distance metric metadata
        self._init_collection()

    def _init_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        # 'hnsw:space' defines the distance metric in ChromaDB
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": self.distance_metric}
        )

    def switch_metric(self, new_metric: str, chunks: Optional[List[DocumentChunk]] = None):
        """Allows switching between Cosine Similarity and Euclidean Distance."""
        if new_metric not in ["cosine", "l2"]:
            raise ValueError(f"Invalid metric {new_metric}")
        self.distance_metric = new_metric
        self._init_collection()
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Indexes DocumentChunk instances into ChromaDB."""
        if not chunks:
            return

        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [c.metadata["chunk_id"] for c in chunks]

        # Chroma batches insertions for efficiency
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top_k relevant chunks with their distance scores.
        Returns list of (DocumentChunk, distance).
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists):
                chunk = DocumentChunk(text=doc, metadata=meta)
                retrieved.append((chunk, float(dist)))

        return retrieved


# Strict Guardrail System Prompt
RAG_SYSTEM_PROMPT = f"""You are an advanced, truthful document QA assistant equipped with STRICT HALLUCINATION GUARDRAILS.

YOUR PRIMARY DIRECTIVE:
1. You must answer the user's question using ONLY the factual evidence provided in the "Context" section below.
2. If the answer cannot be found directly or inferred with 100% certainty from the provided Context, you MUST respond EXACTLY with:
   "{STRICT_GUARDRAIL_FALLBACK}"
3. Do NOT make assumptions, guess, extrapolate, or utilize outside knowledge.
4. If the Context contains the answer, provide a clear, concise, and well-structured response citing relevant page numbers when available.
5. Never invent facts, dates, names, or numbers not present in the Context.
"""


def build_rag_prompt(query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> Tuple[List[Any], str]:
    """
    Constructs the formatted prompt for the LLM including context chunks.
    """
    context_blocks = []
    for idx, (chunk, dist) in enumerate(retrieved_chunks, start=1):
        page = chunk.metadata.get("page", "Unknown")
        source = chunk.metadata.get("source", "PDF")
        context_blocks.append(
            f"--- Context Excerpt {idx} (Source: {source}, Page: {page}, Distance: {dist:.4f}) ---\n{chunk.text}"
        )

    context_str = "\n\n".join(context_blocks)
    user_content = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    return messages, context_str


def generate_rag_answer_stream(
    query: str,
    retrieved_chunks: List[Tuple[DocumentChunk, float]],
    api_key: str,
    model_name: str = DEFAULT_MODEL
) -> Generator[str, None, None]:
    """
    Streams the RAG response from Groq LLM using strict hallucination guardrails.
    """
    if not retrieved_chunks:
        yield STRICT_GUARDRAIL_FALLBACK
        return

    if not api_key:
        yield "Error: Groq API Key is missing. Please configure your GROQ_API_KEY in .env or the sidebar."
        return

    messages, _ = build_rag_prompt(query, retrieved_chunks)

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0,  # Zero temperature for deterministic, factual fidelity
            streaming=True
        )

        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        yield f"Error querying Groq LLM: {str(e)}"
