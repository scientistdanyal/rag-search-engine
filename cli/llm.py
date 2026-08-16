import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

import re
import time
import json
def enhance_query_spell(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def enhance_query_rewrite(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()



def enhance_query_expand(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()



def _get_openrouter_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
def rerank_individual(query: str, documents: list[dict]) -> list[dict]:
    client = _get_openrouter_client()
    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.
Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}
Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness
Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.
Score:"""
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
        )
        content = (response.choices[0].message.content or "").strip()
        match = re.search(r"\d+(?:\.\d+)?", content)
        doc["llm_score"] = float(match.group()) if match else 0.0
        time.sleep(3)  # avoid rate limits
    return sorted(documents, key=lambda d: d["llm_score"], reverse=True)




def rerank_batch(query: str, documents: list[dict]) -> list[dict]:
    client = _get_openrouter_client()


    doc_lines = []
    for doc in documents:
        doc_id = doc.get("id")
        title = doc.get("title", "")
        document = doc.get("document", "")
        doc_lines.append(f"ID: {doc_id} | {title} | {document}")
        
    doc_list_str = "\n".join(doc_lines)

    prompt = f"""Rank the movies listed below by relevance to the following search query.
Query: "{query}"
Movies:
{doc_list_str}
Return the movie IDs in order of relevance, best match first.
Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.
For example:
[75, 12, 34, 2, 1]
Ranking:"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("josn"):
            content = content[4:].strip()
    ranked_ids = json.loads(content)


    by_id = {int(doc["id"]): doc for doc in documents}
    ranked_results = []
    for rank, movie_id in enumerate(ranked_ids, start=1):
        movie_id = int(movie_id)
        if movie_id not in by_id:
            continue
        doc = by_id[movie_id]
        doc["llm_rank"] = rank
        ranked_results.append(doc)
    # Keep any docs the model omitted, at the end
    seen = {int(d["id"]) for d in ranked_results}
    for doc in documents:
        if int(doc["id"]) not in seen:
            doc["llm_rank"] = len(ranked_results) + 1
            ranked_results.append(doc)
    return sorted(ranked_results, key=lambda d: d["llm_rank"])


def rerank_cross_encoder(query: str, documents: list[dict]) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append(
            [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        )
    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    except Exception:
        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L2-v2",
            device="cpu",
        )
    scores = cross_encoder.predict(pairs)
    for doc, score in zip(documents, scores):
        doc["cross_encoder_score"] = float(score)
    return sorted(
        documents,
        key=lambda d: d["cross_encoder_score"],
        reverse=True,
    )