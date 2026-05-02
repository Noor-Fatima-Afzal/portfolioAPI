from __future__ import annotations

import os
import re
from datetime import date
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    relevance_score: float = 0.0


class CVRAGPipeline:
    def __init__(self) -> None:
        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.cv_path = os.getenv("CV_PATH", "Noor_Fatima_CV.pdf")

        self._chunks: List[Chunk] = []
        self._full_text: str = ""
        self._lines: List[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

        self._client = Groq(api_key=self.api_key) if self.api_key else None
        self._build_knowledge_base()

    @property
    def is_ready(self) -> bool:
        return self._client is not None and len(self._chunks) > 0 and self._matrix is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _read_cv_text(self) -> str:
        if not os.path.exists(self.cv_path):
            raise FileNotFoundError(
                f"CV file not found at '{self.cv_path}'. Update CV_PATH in .env to your CV file path."
            )

        reader = PdfReader(self.cv_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages).strip()

        if not full_text:
            raise ValueError("Could not extract any text from the CV PDF.")

        return full_text

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap_chars: int = 180) -> List[Chunk]:
        # Build paragraph-aware chunks to preserve CV sections and line-level facts.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(Chunk(text=current))

                # Add a short tail overlap to improve retrieval continuity between chunks.
                overlap = current[-overlap_chars:].strip()
                current = f"{overlap}\n{para}".strip() if overlap else para
            else:
                # Very long paragraph fallback.
                chunks.append(Chunk(text=para[:chunk_size].strip()))
                current = para[chunk_size - overlap_chars :].strip()

        if current:
            chunks.append(Chunk(text=current))

        return chunks

    def _question_keywords(self, question: str) -> List[str]:
        base_tokens = re.findall(r"[a-zA-Z]{3,}", question.lower())

        boosters: List[str] = []
        q = question.lower()
        if any(k in q for k in ["graduat", "study", "education", "degree", "university", "college"]):
            boosters.extend(["education", "degree", "bachelor", "master", "university", "college", "graduation", "student", "cgpa", "gpa"])
        if any(k in q for k in ["experience", "machine learning", "ml", "ai", "years"]):
            boosters.extend(["experience", "machine", "learning", "ml", "ai", "internship", "project", "research", "fellowship", "year", "month"])

        unique_tokens: List[str] = []
        for token in base_tokens + boosters:
            if token not in unique_tokens:
                unique_tokens.append(token)
        return unique_tokens

    def _keyword_line_context(self, question: str, max_lines: int = 18) -> List[str]:
        if not self._lines:
            return []

        keywords = self._question_keywords(question)
        if not keywords:
            return []

        matched_indices: List[int] = []
        for idx, line in enumerate(self._lines):
            low = line.lower()
            if any(k in low for k in keywords):
                matched_indices.append(idx)

        if not matched_indices:
            return []

        # Include neighboring lines to capture values on the next/previous line.
        selected: List[str] = []
        seen: set[int] = set()
        for idx in matched_indices[: max_lines // 2]:
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < len(self._lines) and j not in seen:
                    seen.add(j)
                    selected.append(self._lines[j])
                if len(selected) >= max_lines:
                    return selected
        return selected

    def _find_education_lines(self) -> List[str]:
        edu_keys = ["education", "degree", "bachelor", "master", "bs", "ms", "university", "college", "student", "graduat"]
        lines: List[str] = []
        for line in self._lines:
            low = line.lower()
            if any(k in low for k in edu_keys):
                lines.append(line)
        return lines

    def _extract_studying_or_graduation(self, question: str) -> str | None:
        if not self._lines:
            return None

        q = question.lower()
        wants_graduation = any(k in q for k in ["graduat", "pass out", "completion", "complete degree", "graduation year"])
        wants_study = any(k in q for k in ["study", "studying", "student", "degree", "education", "major", "field"])

        if not (wants_graduation or wants_study):
            return None

        edu_lines = self._find_education_lines()
        if not edu_lines:
            return None

        year_re = re.compile(r"\b(19\d{2}|20\d{2})\b")

        def _normalize_line(line: str) -> str:
            line = re.sub(r"([A-Za-z])(\d{4})", r"\1 \2", line)
            line = re.sub(r"\s+", " ", line).strip()
            return line

        if wants_graduation:
            for line in edu_lines:
                clean = _normalize_line(line)
                years = year_re.findall(clean)
                low = clean.lower()
                if years and any(k in low for k in ["graduat", "degree", "bachelor", "master", "bs", "ms", "expected"]):
                    expected_match = re.search(r"expected\s*(?:in\s*)?(19\d{2}|20\d{2})", low, re.IGNORECASE)
                    if expected_match:
                        return f"She is expected to graduate in {expected_match.group(1)}."

                    if len(years) >= 2:
                        start_year = years[0]
                        end_year = years[-1]
                        return f"She started her degree in {start_year} and is expected to graduate in {end_year}."

                    return f"She is expected to graduate in {years[-1]}."

            for i, line in enumerate(self._lines):
                low = line.lower()
                if "graduat" in low:
                    window = [self._lines[j] for j in range(max(0, i - 1), min(len(self._lines), i + 2))]
                    text = _normalize_line(" | ".join(window))
                    if year_re.search(text):
                        expected_match = re.search(r"expected\s*(?:in\s*)?(19\d{2}|20\d{2})", text, re.IGNORECASE)
                        if expected_match:
                            return f"She is expected to graduate in {expected_match.group(1)}."

                        years = year_re.findall(text)
                        if len(years) >= 2:
                            return f"She started her degree in {years[0]} and is expected to graduate in {years[-1]}."
                        return f"She is expected to graduate in {years[-1]}."

        if wants_study:
            for line in edu_lines:
                clean = _normalize_line(line)
                low = clean.lower()
                if any(k in low for k in ["currently", "student", "studying", "bachelor", "master", "bs", "ms"]):
                    years = year_re.findall(clean)
                    if len(years) >= 2:
                        return f"She is currently pursuing {clean.split(str(years[0]))[0].strip()} ({years[0]} to {years[-1]})."
                    return f"She is currently pursuing {clean}."

            # Fallback to the strongest education line.
            return f"She is currently pursuing {_normalize_line(edu_lines[0])}."

        return None

    def _extract_ml_experience(self, question: str) -> str | None:
        q = question.lower()
        asks_ml_exp = any(k in q for k in ["machine learning", "ml", "ai"]) and any(
            k in q for k in ["experience", "experince", "expereince", "year", "years", "duration", "total"]
        )
        if not asks_ml_exp:
            return None

        if not self._lines:
            return None

        duration_re = re.compile(r"\b(\d+)\s*\+?\s*(year|years|month|months)\b", re.IGNORECASE)

        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month_names = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        date_range_re = re.compile(
            rf"\b(?P<smon>{month_names})\s*(?P<syear>20\d{{2}}|19\d{{2}})\s*[\-–—]\s*(?:(?P<emon>{month_names})\s*(?P<eyear>20\d{{2}}|19\d{{2}})|(?P<present>present|current|ongoing|now))",
            re.IGNORECASE,
        )

        def _normalize_line(line: str) -> str:
            # Handle PDF extraction issues like "InternMar 2024".
            line = re.sub(rf"([A-Za-z])({month_names})", r"\1 \2", line, flags=re.IGNORECASE)
            line = re.sub(r"\s+", " ", line).strip()
            return line

        def _to_month_number(month_text: str) -> int:
            return month_map[month_text[:3].lower()]

        def _months_between(start_date: date, end_date: date) -> int:
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if end_date.day < start_date.day:
                months -= 1
            return max(months, 0)

        def _format_duration(month_count: int) -> str:
            years = month_count // 12
            months = month_count % 12
            year_label = "year" if years == 1 else "years"
            month_label = "month" if months == 1 else "months"

            if years > 0 and months > 0:
                return f"{years} {year_label} and {months} {month_label}"
            if years > 0:
                return f"{years} {year_label}"
            return f"{months} {month_label}"

        ml_indices = [
            i
            for i, line in enumerate(self._lines)
            if any(k in line.lower() for k in ["machine learning", "ml", "artificial intelligence", "ai", "deep learning"]) 
        ]

        if not ml_indices:
            return None

        total_months = 0
        evidence: List[str] = []
        ml_start_dates: List[date] = []
        seen_windows: set[int] = set()

        for idx in ml_indices:
            for j in range(max(0, idx - 1), min(len(self._lines), idx + 2)):
                if j in seen_windows:
                    continue
                seen_windows.add(j)

                line = self._lines[j]
                clean_line = _normalize_line(line)

                for match in date_range_re.finditer(clean_line):
                    start_month = _to_month_number(match.group("smon"))
                    start_year = int(match.group("syear"))
                    ml_start_dates.append(date(start_year, start_month, 1))

                for value, unit in duration_re.findall(line):
                    amount = int(value)
                    if unit.lower().startswith("year"):
                        total_months += amount * 12
                    else:
                        total_months += amount
                if duration_re.search(line) or ("machine learning" in line.lower() and line not in evidence):
                    evidence.append(clean_line)

        if ml_start_dates:
            first_start = min(ml_start_dates)
            today = date.today()
            span_months = _months_between(first_start, today)
            duration_text = _format_duration(span_months)

            start_text = first_start.strftime("%b %Y")
            return (
                "From her first machine-learning role in "
                f"{start_text} to the present, she has about {duration_text} of total machine-learning experience."
            )

        if total_months > 0:
            duration_text = _format_duration(total_months)
            return (
                "Based on explicit durations in the CV, the candidate has at least "
                f"{duration_text} of machine-learning-related experience. "
                f"Relevant CV lines: {' | '.join(evidence[:3])}"
            )

        if evidence:
            return (
                "The CV shows machine-learning-related projects/internships, but it does not provide a clear total duration. "
                f"Relevant CV lines: {' | '.join(evidence[:3])}"
            )

        return None

    def _rule_based_answer(self, question: str) -> str | None:
        answer = self._extract_studying_or_graduation(question)
        if answer:
            return answer

        answer = self._extract_ml_experience(question)
        if answer:
            return answer

        return None

    def _build_knowledge_base(self) -> None:
        text = self._read_cv_text()
        self._full_text = text
        self._lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        self._chunks = self._chunk_text(text)

        if not self._chunks:
            raise ValueError("No chunks could be generated from CV text.")

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        corpus = [chunk.text for chunk in self._chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def _retrieve(self, question: str, top_k: int = 4) -> List[Chunk]:
        """Retrieve relevant chunks with strict relevance thresholds to prevent hallucinations."""
        if self._vectorizer is None or self._matrix is None:
            return []

        question_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(question_vec, self._matrix).flatten()

        if scores.size == 0:
            return []

        top_indices = scores.argsort()[::-1][:top_k]
        
        # STRICT: Only return chunks with meaningful relevance (>0.05 cosine similarity)
        # This prevents returning random chunks that could cause hallucinations
        MIN_RELEVANCE_THRESHOLD = 0.05
        relevant_chunks = []
        
        for idx in top_indices:
            if scores[idx] >= MIN_RELEVANCE_THRESHOLD:
                chunk = self._chunks[idx]
                chunk.relevance_score = float(scores[idx])
                relevant_chunks.append(chunk)
        
        # If no sufficiently relevant chunks found, return empty to trigger explicit "don't know" response
        return relevant_chunks

    def answer_question(self, question: str, top_k: int = 4) -> Tuple[str, int]:
        if not self._client:
            raise RuntimeError("Missing GROQ_API_KEY in .env")

        # Try rule-based extraction first (most reliable)
        rule_answer = self._rule_based_answer(question)
        if rule_answer:
            return rule_answer, 1

        effective_top_k = min(max(top_k, 4), 8)
        retrieved_chunks = self._retrieve(question, top_k=effective_top_k)
        keyword_lines = self._keyword_line_context(question)

        # STRICT: Check if we have meaningful context. If not, return "don't know" immediately
        has_strong_context = (
            len(retrieved_chunks) > 0 or 
            len(keyword_lines) > 0
        )
        
        if not has_strong_context:
            return (
                "I don't have information about that in the CV. "
                "Would you like to ask the candidate directly or explore other topics from the CV?",
                0
            )

        context_parts: List[str] = []
        if retrieved_chunks:
            context_parts.append(
                "Retrieved CV Chunks:\n" + "\n\n".join([f"- {c.text}" for c in retrieved_chunks])
            )
        if keyword_lines:
            context_parts.append(
                "Keyword-Matched CV Lines:\n" + "\n".join([f"- {line}" for line in keyword_lines])
            )

        context = "\n\n".join(context_parts)

        # STRICT: Strongly enforce factual accuracy, refuse hallucinations, require evidence
        system_prompt = (
            "You are a factual and professional portfolio assistant. "
            "Your PRIMARY responsibility is accuracy - NEVER invent, assume, or guess information. "
            "\n\nRULES (CRITICAL - FOLLOW STRICTLY):"
            "\n1. Answer ONLY using explicit facts from the provided CV context. Do not infer or assume."
            "\n2. If information is NOT in the context, say 'I don't have that information' - NEVER fabricate details."
            "\n3. For every claim you make, it MUST come directly from the provided CV text."
            "\n4. If the CV is incomplete or ambiguous, acknowledge the limitation explicitly."
            "\n5. Never say 'based on the CV' unless you're directly quoting or closely paraphrasing the context."
            "\n6. If asked about something outside the CV scope, politely decline and refocus on CV content."
            "\n7. Do not fill gaps with assumptions - if a date, number, or detail is missing, say so explicitly."
            "\n8. Keep responses concise and warm while being strictly factual."
            "\n9. If you must say something isn't available, be direct: 'This isn't mentioned in the CV.'"
            "\n\nRemember: Accuracy over helpfulness. A 'I don't know' is better than a wrong answer."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            "CV Context:\n"
            f"{context}\n\n"
            "Using ONLY the context above, answer the question. "
            "If the context doesn't contain the answer, explicitly say so."
        )

        completion = self._client.chat.completions.create(
            model=self.model_name,
            temperature=0.05,  # Ultra-low temp for maximum consistency and less speculation
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = completion.choices[0].message.content or "I couldn't generate a response. Please try again."
        source_count = len(retrieved_chunks) + (1 if keyword_lines else 0)
        return answer.strip(), source_count
