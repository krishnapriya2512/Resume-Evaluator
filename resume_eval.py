""" 
Implement the following in streamlit using gemini API - 
🔹 1. Bias & Fairness Check Detect gendered language 
    👉 Unique angle: Combines your bias detection project with resume evaluation. 
🔹 2. ATS (Applicant Tracking System) Friendliness Score 
    👉 Evaluate how well the resume passes through ATS filters. Check keyword density vs. job description. Highlight missing ATS-friendly sections (skills, certifications, experience). 
🔹 3. Soft Skills vs. Hard Skills Balance 
    👉 Automatically detect soft skills (leadership, communication) vs hard skills (Python, TensorFlow). Give feedback if one is underrepresented. 
🔹 4. Quantification & Impact Detection 
    👉 Highlight sentences without measurable impact. Example: "Worked on sales" → suggest: "Increased sales by 30% within 6 months". Adds personalized improvement suggestions. 
🔹 5. Readability & Tone Analysis 
    👉 Use NLP to check clarity, conciseness, and tone. Give a Readability Score (Flesch-Kincaid). Suggest improvements like “avoid passive voice”, “make sentences shorter”. 
🔹 6. Job Role Alignment Insights 
    👉 Instead of just match %, show “Top 5 skills missing for this role”. Give career path suggestions (“This resume is closer to Data Analyst roles than Data Scientist roles”). 
🔹 7. Resume Comparison Mode 
    👉 Upload two resumes (e.g., old vs new, or your resume vs a sample industry resume). Get a side-by-side comparison with suggestions. 
🔹 8. Personalized Learning Resources For missing skills, recommend courses, certifications, or projects (Coursera, LinkedIn Learning, Kaggle). 
    👉 Example: “You’re missing SQL for Data Analyst role. Suggested course: ‘SQL for Data Science’ on Coursera.” 
🔹 9. Diversity & Inclusive Language Suggestions Flag language that might be outdated or biased.
    👉 Suggest inclusive alternatives (e.g., “chairman” → “chairperson”). 
🔹 10. Career Level Adaptation Detect if the resume is entry-level, mid-level, or senior-level. 
    👉 Adjust feedback accordingly (e.g., “Add internships” for students, “Highlight leadership impact” for seniors). 
🔹 11. Visual Resume Score (Optional if PDF Parsing) 
    👉 Evaluate formatting, section organization, bullet clarity, whitespace usage. Provide a Design Score along with text feedback. 
🔹 12. Interview Readiness Insights
"""

# Importing necessary libraries
import io
import re
import json
import base64
import time
import pdf2image
import pandas as pd
import textstat
import spacy
import streamlit as st
import google.generativeai as genai
from collections import Counter


import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


# Gemini response function -------------------------------------------

def get_gemini_response(input_prompt, pdf_content, job_desc, model_name='gemini-2.5-flash'):
    """
    Sends 3-part input to gemini as you already did (input_prompt, pdf image bytes encoded, job_desc).
    Returns response.text (string).
    """
    model = genai.GenerativeModel(model_name)
    # keep your 3-item array style so earlier behavior is preserved
    response = model.generate_content([input_prompt, pdf_content[0], job_desc])
    return response.text


# Defining PDF -> image setup --------------------------------------

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        poppler_path = r"C:\Program Files (x86)\poppler\Library\bin"
        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [
            {
                "mime_type": 'image/jpeg',
                "data": base64.b64encode(img_byte_arr).decode(),
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")


# JSON parsing helper ----------------------------------------------

def parse_gemini_json(raw_text):
    cleaned = raw_text.strip()
    # remove starting/ending code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # In many LLM outputs there might be trailing text; try to find JSON substring
    json_like = None
    # find first { and last }
    first_curly = cleaned.find("{")
    last_curly = cleaned.rfind("}")
    if first_curly != -1 and last_curly != -1 and last_curly > first_curly:
        json_like = cleaned[first_curly:last_curly+1]
    else:
        json_like = cleaned  # fallback, might still fail
    try:
        data = json.loads(json_like)
        return data
    except Exception:
        # final fallback: try to evaluate as list
        try:
            return json.loads(cleaned)
        except Exception:
            return None


# Small lexicons & helper utilities ----------------------------------------------

HARD_SKILLS = {"python", "sql", "tensorflow", "pytorch", "aws", "spark", "hadoop", "scala", "docker", "kubernetes", "excel", "r"}
SOFT_SKILLS = {"leadership", "communication", "teamwork", "collaboration", "problem solving", "time management", "adaptability", "creativity"}
INCLUSIVE_MAP = {"chairman":"chairperson","mankind":"humankind","manpower":"workforce","salesman":"salesperson","fireman":"firefighter","policeman":"police officer"}

def text_to_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_skill_matches(text):
    t = text.lower()
    found_hard = {s for s in HARD_SKILLS if s in t}
    found_soft = {s for s in SOFT_SKILLS if s in t}
    return list(found_hard), list(found_soft)

def count_pronouns_and_gendered_words(text):
    t = text.lower()
    pronouns = {
        "he/him": t.count(" he ") + t.count(" him "),
        "she/her": t.count(" she ") + t.count(" her "),
        "they/them": t.count(" they ") + t.count(" them "),
    }
    # simple gendered adjectives list 
    gendered_adjs = ["aggressive","nurturing","emotional","assertive","caring","dominant","supportive"]
    adjs_found = [a for a in gendered_adjs if a in t]
    return pronouns, adjs_found

def detect_numbers_in_sentence(sent):
    # simple numeric detection
    return bool(re.search(r"\d+%?|\d+\.\d+|one|two|three|four|five|six|seven|eight|nine|ten", sent.lower()))


# Feature 1: Bias & Fairness Check -----------------------------------

def bias_and_fairness_check(resume_content, job_desc):
    """
    1) Local heuristics: pronoun counts and gendered adjectives
    2) Call Gemini to label sentences (batch) and request rewrites for flagged ones
    """
    text = resume_content_to_text(resume_content)
    sentences = text_to_sentences(text)
    pronouns, adjs_found = count_pronouns_and_gendered_words(text)

    # Preparing Gemini prompt to classify sentences in batch (send up to ~30 sentences in one call)
    to_check = sentences[:30]  # limit size for cost/perf
    prompt = (
        "You are a bias-detection assistant. For each resume sentence provide JSON list entries "
        "with fields: sentence, label where label ∈ {Neutral, Gendered-language, Stereotype, Offensive}, "
        "and suggestion (a gender-neutral or stereotype-free rewrite if flagged). Return a JSON array."
        "\n\nSentences:\n"
    )
    for s in to_check:
        prompt += f"- {s}\n"
    gem_response = get_gemini_response(prompt, resume_content, job_desc)
    parsed = parse_gemini_json(gem_response)
    # fallback if parsing fails: ask Gemini for a simple summary
    if parsed is None:
        # create a summarised fallback
        summary_prompt = "Summarize any potential gendered language or stereotypes in the following resume. Return a short JSON: { 'issues': [..], 'recommendation': '...' }"
        gem_response2 = get_gemini_response(summary_prompt, resume_content, job_desc)
        parsed = parse_gemini_json(gem_response2) or {"issues": [], "recommendation": gem_response2}

    result = {
        "pronouns": pronouns,
        "gendered_adjectives_found": adjs_found,
        "gemini_labels": parsed
    }
    return result


# Feature 2: ATS Friendliness Score -----------------------------------

def ats_friendliness_score(resume_content, job_desc):
    """
    Heuristic ATS score:
    - presence of required sections
    - keyword match ratio
    - plain-text vs image-only detection (we already send image; but we can detect if text extraction low)
    """
    text = resume_content_to_text(resume_content).lower()
    score = 0
    reasons = []

    # Section checks
    sections = {"experience": ("experience", "work history", "professional experience"),
                "education": ("education","academic"),
                "skills": ("skills", "technical skills", "expertise")}
    section_found = {}
    for k, variants in sections.items():
        found = any(v in text for v in variants)
        section_found[k] = found
        if found:
            score += 10
        else:
            reasons.append(f"Missing section: {k.capitalize()}")

    # Contact info heuristic
    if re.search(r"\b@\b|mailto:|phone|tel:|\b\d{10}\b", text):
        score += 10
    else:
        reasons.append("Missing explicit contact info text (email/phone)")

    # Keyword density vs JD (simple TF approach)
    jd = job_desc.lower() if job_desc else ""
    if jd:
        # extract keywords by splitting and counting nouns/important words via spaCy
        jd_doc = nlp(jd)
        jd_keywords = [chunk.text.lower() for chunk in jd_doc.noun_chunks if len(chunk.text.split())<=3]
        # count unique matches
        matches = sum(1 for kw in set(jd_keywords) if kw in text)
        total = max(1, len(set(jd_keywords)))
        kw_ratio = matches / total
        score += int(30 * kw_ratio)  # up to 30 points from keywords
        reasons.append(f"Keyword match ratio: {kw_ratio:.2f}")
    else:
        reasons.append("No job description provided; keyword match skipped")

    # readability heuristic (if text is very low, likely image-only resume which is bad for ATS)
    try:
        flesch = textstat.flesch_reading_ease(text)
        if flesch >= 50:
            score += 15
        elif flesch >= 30:
            score += 8
        else:
            score += 3
        reasons.append(f"Flesch score: {flesch:.1f}")
    except Exception:
        reasons.append("Flesch calculation failed")

    # cap score 0-100
    score = max(0, min(100, score))
    return {"ats_score": score, "reasons": reasons, "sections": section_found}


# Feature 3: Soft vs Hard Skills Balance -----------------------------------

def soft_hard_balance(resume_content):
    text = resume_content_to_text(resume_content)
    hard, soft = extract_skill_matches(text)
    total = len(hard) + len(soft)
    balance = {"hard_count": len(hard), "soft_count": len(soft), "hard_skills": hard, "soft_skills": soft}
    if total == 0:
        balance["balance_msg"] = "No skills detected from the predefined lexicon. Consider adding explicit skills section."
    else:
        if len(hard) > len(soft) * 2:
            balance["balance_msg"] = "Hard-skills heavy: consider adding evidence of teamwork/leadership/communication."
        elif len(soft) > len(hard) * 2:
            balance["balance_msg"] = "Soft-skills heavy: add technical keywords and tools if relevant."
        else:
            balance["balance_msg"] = "Balanced mix of soft and hard skills."
    return balance


# Feature 4: Quantification & Impact Detection -----------------------------------

def quantification_and_impact(resume_content, job_desc):
    text = resume_content_to_text(resume_content)
    sentences = text_to_sentences(text)
    to_suggest = []
    for s in sentences:
        # if sentence has action verb-like words but no numbers -> candidate for quantification
        if re.search(r"\b(managed|led|increased|decreased|reduced|improved|developed|built|created|optimized|boosted|grew)\b", s.lower()):
            if not detect_numbers_in_sentence(s):
                to_suggest.append(s)
    # Batch ask Gemini for suggested quantifiable rewrites/templates
    suggestions = []
    if to_suggest:
        prompt = "You are an expert resume coach. For each input bullet, if it lacks measurable impact, provide:\n"
        prompt += " - a suggested quantified rewrite using a plausible template (use X%/N placeholders if exact numbers unknown)\n"
        prompt += "Return a JSON array with { 'original':..., 'suggestion':... }\n\nBullets:\n"
        for b in to_suggest[:30]:
            prompt += f"- {b}\n"
        gem_resp = get_gemini_response(prompt, resume_content, job_desc)
        parsed = parse_gemini_json(gem_resp)
        if parsed is None:
            # fallback: simple template generation locally
            parsed = []
            for b in to_suggest:
                parsed.append({"original": b, "suggestion": b + " (e.g., increased X% within Y months — add numbers)"})
        suggestions = parsed
    return {"candidates": to_suggest, "suggestions": suggestions}


# Feature 5: Readability & Tone Analysis -----------------------------------

def readability_and_tone(resume_content, job_desc):
    text = resume_content_to_text(resume_content)
    flesch = None
    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = None
    # passive voice detection (simple heuristic through spaCy dependencies)
    doc = nlp(text)
    passive_sents = []
    for sent in doc.sents:
        if any(tok.dep_ == "auxpass" for tok in sent):
            passive_sents.append(sent.text.strip())
    # Use Gemini to provide short rewrite suggestions for long or passive sentences
    long_sentences = [s.text.strip() for s in doc.sents if len(s.text.split()) > 30][:10]
    gem_suggestions = []
    if long_sentences:
        prompt = "Make the following resume sentences more concise and active. Return JSON [{original:'', rewrite:''},...]\nSentences:\n"
        for s in long_sentences:
            prompt += f"- {s}\n"
        gem_resp = get_gemini_response(prompt, resume_content, job_desc)
        parsed = parse_gemini_json(gem_resp)
        if parsed:
            gem_suggestions = parsed
    return {"flesch": flesch, "passive_sentences": passive_sents, "concise_suggestions": gem_suggestions}


# Feature 6: Job Role Alignment Insights -----------------------------------

# small role templates (extendable or load from JSON)
ROLE_SKILLS = {
    "data scientist": ["python","sql","machine learning","statistics","tensorflow","modeling","visualization"],
    "data analyst": ["sql","excel","visualization","tableau","reporting","python","statistics"],
    "ml engineer": ["python","pytorch","tensorflow","docker","kubernetes","model deployment","api"]
}

def role_alignment_insights(resume_content, job_desc, top_role="data scientist"):
    text = resume_content_to_text(resume_content)
    resume_skills = set(extract_skill_matches(text)[0] + extract_skill_matches(text)[1])
    role_skills = set(ROLE_SKILLS.get(top_role.lower(), []))
    missing = list(role_skills - resume_skills)
    overlap = list(role_skills & resume_skills)
    # Ask Gemini to explain alignment briefly
    prompt = f"Explain briefly (1-2 sentences) whether the resume matches the role '{top_role}' given these overlaps: {overlap} and missing: {missing}. Output JSON {{'role':'{top_role}','alignment':'short sentence'}}"
    gem_resp = get_gemini_response(prompt, resume_content, job_desc)
    parsed = parse_gemini_json(gem_resp) or {"role": top_role, "alignment": gem_resp}
    return {"role": top_role, "missing_skills": missing, "overlap_skills": overlap, "explanation": parsed.get("alignment", "")}


# Feature 7: Resume Comparison Mode -------------------------------------

def compare_two_resumes(resume_content_a, resume_content_b, job_desc):
    text_a = resume_content_to_text(resume_content_a)
    text_b = resume_content_to_text(resume_content_b)
    hard_a, soft_a = extract_skill_matches(text_a)
    hard_b, soft_b = extract_skill_matches(text_b)
    # simple match percentages vs job desc
    ats_a = ats_friendliness_score(resume_content_a, job_desc)['ats_score']
    ats_b = ats_friendliness_score(resume_content_b, job_desc)['ats_score']
    return {
        "resume_a": {"ats": ats_a, "hard": hard_a, "soft": soft_a},
        "resume_b": {"ats": ats_b, "hard": hard_b, "soft": soft_b},
    }


# Feature 8: Personalized Learning Resources (local curated map) -----------------------------

RESOURCE_MAP = {
    "sql": ["Coursera - SQL for Data Science", "Mode SQL tutorial"],
    "python": ["Coursera - Python for Everybody", "Kaggle - Python courses"],
    "tensorflow": ["Coursera - TensorFlow in Practice", "TensorFlow official tutorials"],
    "tableau": ["Udemy - Tableau A-Z", "Tableau official training"]
}

def recommend_learning_resources(missing_skills):
    recs = {}
    for s in missing_skills:
        key = s.lower()
        if key in RESOURCE_MAP:
            recs[s] = RESOURCE_MAP[key]
        else:
            recs[s] = ["Search Coursera/LinkedIn Learning for " + s]
    return recs


# Feature 9: Diversity & Inclusive Language Suggestions ------------------------------------

def inclusive_language_suggestions(resume_content):
    text = resume_content_to_text(resume_content)
    suggestions = []
    for old, new in INCLUSIVE_MAP.items():
        if re.search(r"\b" + re.escape(old) + r"\b", text, flags=re.IGNORECASE):
            suggestions.append({"original": old, "suggestion": new})
    return suggestions


# Feature 10: Career Level Adaptation ------------------------------------

def detect_career_level(resume_content):
    text = resume_content_to_text(resume_content)
    # years heuristic: find years like "2018 - 2022" and calculate
    years = re.findall(r"(19|20)\d{2}", text)
    years = [int(y) for y in years]
    exp_years_est = 0
    if years:
        exp_years_est = max(years) - min(years)
    # leadership mentions
    leadership_mentions = len(re.findall(r"\b(managed|led|supervised|head|director|owner)\b", text.lower()))
    if exp_years_est < 2 and leadership_mentions == 0:
        level = "Entry-level"
    elif exp_years_est < 6:
        level = "Mid-level"
    else:
        level = "Senior-level"
    return {"estimated_years_range": exp_years_est, "leadership_mentions": leadership_mentions, "level": level}


# Feature 11: Visual Resume Score (basic heuristics) -----------------------------------

def visual_resume_score_from_image(pdf_content):
    # We only have the first page image; perform lightweight heuristics:
    # - If we have an image, we assume formatting may be present. We'll keep a simple rule-based score.
    if not pdf_content:
        return {"design_score": 0, "notes": ["No PDF content provided"]}
    # heuristics:
    score = 60
    notes = []
    # penalize if content is an image-only resume (we can't extract text easily)
    # (we already extracted image; but we can attempt OCR - skip for now to keep dependencies small)
    notes.append("Design heuristics used: standard margins, bullets, headings assumed.")
    return {"design_score": score, "notes": notes}


# Feature 12: Interview Readiness Insights --------------------------------------

def interview_questions_from_resume(resume_content, job_desc):
    text = resume_content_to_text(resume_content)
    # Choose 8-12 important bullets (simple heuristic: sentences with keywords)
    candidates = []
    for s in text_to_sentences(text):
        if re.search(r"\b(model|algorithm|managed|led|built|deployed|improved|reduced|increased|designed)\b", s.lower()):
            candidates.append(s)
    candidates = candidates[:10]
    questions = []
    if candidates:
        prompt = "For each resume bullet below, generate 2 technical and 1 behavioral interview question. Return JSON [{ 'bullet':..., 'technical': ['..','..'], 'behavioral': '...'}]\nBullets:\n"
        for c in candidates:
            prompt += f"- {c}\n"
        gem_resp = get_gemini_response(prompt, resume_content, job_desc)
        parsed = parse_gemini_json(gem_resp)
        if parsed:
            questions = parsed
        else:
            # fallback pseudo-questions
            for c in candidates:
                questions.append({"bullet": c, "technical": ["What algorithm did you use and why?","How did you evaluate performance?"], "behavioral": "Tell me about a challenge you faced during this project."})
    return questions

# -------------------------
# Helper: convert resume pdf_content (image) back to lightweight text placeholder
# (In your code you pass pdf_content to Gemini which likely performs OCR. For local heuristics we need some text.)
# If you already extract full resume text elsewhere, connect that here. For now we ask Gemini for extracted text if needed.
# -------------------------
def resume_content_to_text(pdf_content):
    """
    Try to get plain text from the resume via a Gemini call (single-shot). This keeps your existing pattern: we send image to gemini and ask for extracted text.
    """
    # prompt to extract text from provided image content
    prompt = "Extract the resume text from the provided image and return only the plain text (no commentary)."
    try:
        gem_resp = get_gemini_response(prompt, pdf_content, "")
        # gem_resp is likely plain text
        return gem_resp
    except Exception:
        return ""


# UI: Streamlit components -------------------------------------

st.header("Resume Evaluator")
input_text = st.text_area("Enter the Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your Resume (PDF): ", type=["pdf"], key="file_uploader")
uploaded_file2 = st.file_uploader("(Optional) Upload second Resume to compare (PDF): ", type=["pdf"], key="file_uploader_2")

if uploaded_file is not None:
    st.success("Primary file uploaded successfully!")

# Buttons for actions (kept originals + added combined analysis)
col1, col2 = st.columns(2)
with col1:
    submit1 = st.button("Percentage Match with Job Description")
    submit2 = st.button("Matched Skills")
    submit3 = st.button("Evaluate Detailed Metrics")
    submit4 = st.button("How Can I Improve My Resume?")
with col2:
    submit_bias = st.button("Bias & Fairness Check")
    submit_ats = st.button("ATS Friendliness Score")
    submit_quant = st.button("Quantification Suggestions")
    submit_interview = st.button("Interview Questions")

submit_compare = st.button("Compare with second resume")
submit_full = st.button("Run Full Analysis (All Features)")


# Handler: run each request ---------------------------------

if uploaded_file is None:
    st.info("Please upload a PDF file to analyze.")
else:
    pdf_content = input_pdf_setup(uploaded_file)

    input_prompt1 = """
    You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of any one job role from Data Science, Artificial Intelligence, Machine Learning, Data Analysis, Big Data Engineering and deep ATS functionality, 
    your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
    the job description. First the output should come as percentage and then keywords missing and last final thoughts.
    """
    input_prompt2 = """
    You are an expert in identifying key skills and competencies for various job roles in the fields of Data Science, Artificial Intelligence, Machine Learning, Data Analysis, and Big Data Engineering.
    Your task is to analyze the provided resume and extract the skills that match the job description.
    Please provide a list of matched skills along with their relevance to the job role.
    """
    input_prompt3 = """
    You are an experienced Human Resource Manager with Technical Experience in the field of any one job role from Data Science, Artificial Intelligence, Machine Learning, Data Analysis, Big Data Engineering. 
    Your task is to review the provided resume against the job description for these profiles. 
    Please share your professional evaluation on whether the candidate's profile aligns with the role. 
    Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    """

    def evaluate_resume_metrics(job_desc, resume_content):
        prompts = {
            "tone": f"""
            You are an ATS evaluation assistant. 
            Rate the tone of this resume from 0 to 10, where:
            0 = extremely unprofessional, 10 = highly professional, confident, and uses active voice.
            Output ONLY in the following JSON format: {{ "tone_score": <number>, "reason": "<short explanation>" }}.
            """,

            "relevance": f"""
            You are an ATS evaluation assistant.
            Rate the relevance of this resume to the job description from 0 to 10, where:
            0 = no match at all, 10 = perfect alignment with the job description.
            Output ONLY in JSON format: {{ "relevance_score": <number>, "reason": "<short explanation>" }}.
            """,

            "clarity": f"""
            You are an ATS evaluation assistant.
            Rate the clarity of this resume from 0 to 10, where:
            0 = very unclear, poorly structured, difficult to read, 
            10 = extremely clear, well-structured, and easy to read.
            Output ONLY in JSON format: {{ "clarity_score": <number>, "reason": "<short explanation>" }}.
            """,

            "accuracy": f"""
            You are an ATS evaluation assistant.
            Rate the accuracy and specificity of the resume from 0 to 10, where:
            0 = vague and unverified claims, 10 = all claims are specific, measurable, and realistic.
            Output ONLY in JSON format: {{ "accuracy_score": <number>, "reason": "<short explanation>" }}.
            """,

        }
        results = []
        for metric, prompt in prompts.items():
            try:
                raw_response = get_gemini_response(prompt, resume_content, job_desc).strip()

                parsed = parse_gemini_json(raw_response)

                if parsed:
                    # Extract score & reason based on metric
                    score_key = f"{metric}_score"
                    score = parsed.get(score_key)
                    reason = parsed.get("reason", "")
                else:
                    score = None
                    reason = raw_response  # fallback to raw text

                results.append({
                    "Metric": metric.capitalize(),
                    "Score": score,
                    "Reason": reason
                })

            except Exception as e:
                results.append({
                    "Metric": metric.capitalize(),
                    "Score": None,
                    "Reason": str(e)
                })

        return results

    if submit1:
        start_time = time.time()
   
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("Percentage Match with Job Description")
        st.write(response)
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    if submit2:
        start_time = time.time()
        response = get_gemini_response(input_prompt2, pdf_content, input_text)
        st.subheader("Matched Skills")
        st.write(response)
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    if submit3:
        start_time = time.time()
        scores = evaluate_resume_metrics(input_text, pdf_content)
        df = pd.DataFrame(scores)
        st.subheader("Resume Evaluation Metrics (Tone, Relevance, Clarity, Accuracy)")
        st.dataframe(df, use_container_width=True)
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    if submit4:
        start_time = time.time()
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("How Can I Improve My Resume?")
        st.write(response)
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    # New specialized features:
    if submit_bias:
        start_time = time.time()
        bias_res = bias_and_fairness_check(pdf_content, input_text)
        st.subheader("Bias & Fairness Check")
        st.write("Pronoun counts:", bias_res["pronouns"])
        st.write("Gendered adjectives detected:", bias_res["gendered_adjectives_found"])
        st.write("Gemini labels / suggestions (sample):")
        st.write(bias_res["gemini_labels"])
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    if submit_ats:
        start_time = time.time()
        ats_res = ats_friendliness_score(pdf_content, input_text)
        st.subheader("ATS Friendliness Score")
        st.metric("ATS Score", ats_res["ats_score"])
        st.write("Reasons:")
        for r in ats_res["reasons"]:
            st.write("-", r)
        st.write("Sections found:", ats_res["sections"])
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    if submit_quant:
        start_time = time.time()
        quant_res = quantification_and_impact(pdf_content, input_text)
        st.subheader("Quantification & Impact Suggestions")
        if quant_res["candidates"]:
            st.write("Candidates for quantification:")
            for c in quant_res["candidates"]:
                st.write("-", c)
            st.write("Suggested rewrites / templates (Gemini or fallback):")
            st.write(quant_res["suggestions"])
        else:
            st.write("No obvious bullets needing quantification detected.")
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    if submit_interview:
        start_time = time.time()
        qs = interview_questions_from_resume(pdf_content, input_text)
        st.subheader("Interview Questions (auto-generated)")
        for q in qs:
            st.write("Bullet:", q.get("bullet", ""))
            st.write("Technical:", q.get("technical", []))
            st.write("Behavioral:", q.get("behavioral", ""))
            st.write("---")
        st.write(f"Latency: {round(time.time()-start_time,3)}s")

    # Compare two resumes
    if submit_compare:
        if uploaded_file2 is None:
            st.warning("Please upload a second PDF to compare.")
        else:
            pdf_content2 = input_pdf_setup(uploaded_file2)
            cmp = compare_two_resumes(pdf_content, pdf_content2, input_text)
            st.subheader("Resume Comparison (A vs B)")
            st.write("Resume A (primary) vs Resume B (secondary):")
            st.write(cmp)
            st.write("Tip: Use ATS score and missing skills to choose which resume to apply with.")
    
    # Full analysis button for running everything quickly
    if submit_full:
        start_time = time.time()
        # run several features and aggregate into a report dictionary
        report = {}
        report["ats"] = ats_friendliness_score(pdf_content, input_text)
        report["bias"] = bias_and_fairness_check(pdf_content, input_text)
        report["soft_hard"] = soft_hard_balance(pdf_content)
        report["quant"] = quantification_and_impact(pdf_content, input_text)
        report["readability"] = readability_and_tone(pdf_content, input_text)
        report["role_insight"] = role_alignment_insights(pdf_content, input_text, top_role="Data Scientist")
        report["inclusive"] = inclusive_language_suggestions(pdf_content)
        report["career_level"] = detect_career_level(pdf_content)
        report["visual_design"] = visual_resume_score_from_image(pdf_content)
        report["interview_qs"] = interview_questions_from_resume(pdf_content, input_text)

        # present summary
        st.subheader("Full Analysis Summary")
        st.write("ATS Score:", report["ats"]["ats_score"])
        st.write("Bias summary:", report["bias"]["gendered_adjectives_found"])
        st.write("Soft vs Hard skill counts:", report["soft_hard"]["hard_count"], report["soft_hard"]["soft_count"])
        st.write("Top missing skills for Data Scientist:", report["role_insight"]["missing_skills"])
        st.write("Recommended learning resources for missing skills:")
        st.write(recommend_learning_resources(report["role_insight"]["missing_skills"] or []))
        st.write("Design score:", report["visual_design"]["design_score"])
        st.write("Career Level:", report["career_level"]["level"])
        st.write("Interview Questions (sample):")
        for q in report["interview_qs"][:5]:
            st.write(q)
        st.write(f"Total latency for full analysis: {round(time.time()-start_time,3)}s")
