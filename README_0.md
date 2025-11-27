# Python UV Starter

This is a simple Python [uv](https://docs.astral.uv) starter in Firebase Studio.

## Running

```
uv run main.py
```

## Add dependencies

```
uv add ruff
```









حلو جدًا، نطلع من مرحلة الـ Notebooks وندخل في **مرحلة الـ Backend الحقيقي** 😎

رح أبني لك الآن نظام كامل بـ **FastAPI** يربط:

* ✅ SymPy **MathChecker** (الحكم الرياضي الصارم)
* ✅ LLM Feedback Engine (التقييم التعليمي + الشرح)

ويقدّم Endpoint جاهز:

> `POST /grade_equation`
> يأخذ: سؤال + معادلة نموذجية + معادلة الطالب
> يرد: حكم رياضي + تحليل SymPy + Feedback من الـ LLM

---

## 1️⃣ نظرة سريعة على المعماريّة

الملف الأساسي سيكون مثلًا: `main.py` ويحتوي:

* تعريف `MathChecker` (من شغلنا السابق، مبسّط ومُدمَج هنا).
* دوال:

  * `build_llm_feedback_prompt`
  * `llm_feedback`
* نماذج Pydantic للـ request/response.
* FastAPI app + endpoints.

---

## 2️⃣ كود FastAPI متكامل (ملف واحد جاهز: `main.py`)

> انسخ هذا الملف كما هو (مع تعديلات بسيطة على مفتاح الـ API والموديل) ثم شغّل:
> `uvicorn main:app --reload`

```python
# main.py
# ======================================================
# 🚀 Math Grading API (FastAPI + SymPy + LLM)
#  - POST /grade_equation
# ======================================================

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import os
import json
import re

from fastapi import FastAPI
from pydantic import BaseModel

import sympy as sp
from sympy.parsing.latex import parse_latex

# لو ستستخدم OpenAI مثلا:
# pip install openai
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ======================================================
# 1) SymPy-based MathChecker
# ======================================================

def clean_latex(s: str) -> str:
    s = s.strip()
    s = s.replace("$$", "").replace("$", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def looks_like_latex(s: str) -> bool:
    return "\\" in s or "_{" in s or "^{" in s


def normalize_plain_expr(s: str) -> str:
    s = s.strip()
    s = s.replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3")
    s = re.sub(r"\s+", "", s)
    return s


def parse_expression(expr_str: str) -> sp.Expr:
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression")

    # حاول LaTeX أولاً
    if looks_like_latex(expr_str):
        try:
            expr = parse_latex(clean_latex(expr_str))
            return sp.simplify(expr)
        except Exception:
            pass

    # ثم صيغة بسيطة
    expr_str2 = normalize_plain_expr(expr_str)
    expr = sp.sympify(expr_str2)
    return sp.simplify(expr)


def parse_equation(eq_str: str) -> sp.Expr:
    eq_str = eq_str.strip()
    if "=" in eq_str:
        parts = eq_str.split("=")
        lhs_str = "=".join(parts[:-1])
        rhs_str = parts[-1]
    else:
        lhs_str = eq_str
        rhs_str = "0"

    lhs = parse_expression(lhs_str)
    rhs = parse_expression(rhs_str)
    expr = sp.simplify(lhs - rhs)
    return sp.expand(expr)


def compare_terms(teacher_expr: sp.Expr, student_expr: sp.Expr) -> Dict[str, Any]:
    t_terms = sp.Add.make_args(sp.expand(teacher_expr))
    s_terms = sp.Add.make_args(sp.expand(student_expr))

    def term_map(terms):
        m = {}
        for t in terms:
            c, rest = t.as_coeff_Mul()
            m.setdefault(rest, 0)
            m[rest] += c
        return m

    t_map = term_map(t_terms)
    s_map = term_map(s_terms)

    missing = []
    extra = []
    coeff_diff = []

    for rest, t_coeff in t_map.items():
        s_coeff = s_map.get(rest, 0)
        if sp.simplify(s_coeff) == 0:
            missing.append(str(t_coeff * rest))
        elif sp.simplify(t_coeff - s_coeff) != 0:
            coeff_diff.append(
                {
                    "term": str(rest),
                    "teacher_coeff": float(t_coeff),
                    "student_coeff": float(s_coeff),
                }
            )

    for rest, s_coeff in s_map.items():
        t_coeff = t_map.get(rest, 0)
        if sp.simplify(t_coeff) == 0:
            extra.append(str(s_coeff * rest))

    return {
        "missing_terms": missing,
        "extra_terms": extra,
        "coeff_mismatch": coeff_diff,
    }


@dataclass
class CheckResult:
    is_correct: bool
    teacher_expr: sp.Expr
    student_expr: sp.Expr
    diff_expr: sp.Expr
    error_type: Optional[str]
    details: Dict[str, Any]


class MathChecker:
    def check_equation(self, teacher_str: str, student_str: str) -> CheckResult:
        t_expr = parse_equation(teacher_str)
        s_expr = parse_equation(student_str)
        diff = sp.simplify(t_expr - s_expr)

        if sp.simplify(diff) == 0:
            return CheckResult(
                is_correct=True,
                teacher_expr=t_expr,
                student_expr=s_expr,
                diff_expr=diff,
                error_type=None,
                details={},
            )

        term_analysis = compare_terms(t_expr, s_expr)

        error_type = "unknown"
        if term_analysis["missing_terms"] and not term_analysis["extra_terms"]:
            error_type = "missing_terms"
        elif term_analysis["extra_terms"] and not term_analysis["missing_terms"]:
            error_type = "extra_terms"
        elif term_analysis["coeff_mismatch"]:
            error_type = "coefficient_mismatch"

        return CheckResult(
            is_correct=False,
            teacher_expr=t_expr,
            student_expr=s_expr,
            diff_expr=diff,
            error_type=error_type,
            details=term_analysis,
        )


checker = MathChecker()

# ======================================================
# 2) LLM Feedback Prompt Builder
# ======================================================

def build_llm_feedback_prompt(
    teacher_eq: str,
    student_eq: str,
    sympy_result_dict: dict,
    question_text: str = "",
    teacher_steps: Optional[List[str]] = None,
    student_steps: Optional[List[str]] = None,
) -> str:
    teacher_steps = teacher_steps or []
    student_steps = student_steps or []

    sympy_json = json.dumps(sympy_result_dict, ensure_ascii=False, indent=2)

    prompt = f"""
أنت معلم رياضيات خبير لطلاب المرحلة الإعدادية/الثانوية.
مهمتك:
- مقارنة حل الطالب مع الحل النموذجي.
- استخدام نتيجة المحرك الرمزي (SymPy) كمعلومة مساعدة فقط.
- إرجاع التقييم بصيغة JSON فقط.

### نص السؤال (العربي):
{question_text}

### المعادلة النهائية الصحيحة (من المدرّس):
{teacher_eq}

### المعادلة النهائية للطالب:
{student_eq}

### خطوات الحل النموذجي (إن وجدت):
{json.dumps(teacher_steps, ensure_ascii=False, indent=2)}

### خطوات حل الطالب (إن وجدت):
{json.dumps(student_steps, ensure_ascii=False, indent=2)}

### نتيجة SymPy:
{sympy_json}

أرجوك أرجع النتيجة بصيغة JSON فقط، بالشكل التالي:

{{
  "is_correct": <true or false>,
  "score": <number between 0 and 1>,
  "error_type": "<fully_correct | small_algebra_mistake | concept_mistake | incomplete_solution | off_topic>",
  "short_verdict_ar": "<جملة قصيرة عن صحة الحل>",
  "main_error_ar": "<شرح بسيط عن الخطأ الرئيسي>",
  "step_feedback": [
    {{
      "step_index": 0,
      "is_correct": true,
      "comment_ar": "<تعليق على هذه الخطوة بالعربية>"
    }}
  ],
  "suggested_next_question_ar": "<سؤال تدريبي جديد بالعربية>"
}}
"""
    return prompt


def call_llm(prompt: str) -> dict:
    """
    هذه الدالة تحتاج أن تربطها فعليًا بـ LLM الذي تستخدمه.
    هنا أضع شكلًا عامًّا، عدّل حسب مزوّدك (OpenAI / غيره).

    الآن: أضع تنفيذ بسيط "mock" حتى لا يكسر الكود لو ما عندك LLM جاهز.
    """

    # --- لو عندك OpenAI، استعمل شيء شبيه بهذا:
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "أنت معلم رياضيات خبير ودقيق."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.2,
    # )
    # content = response.choices[0].message.content

    # في هذه النسخة، نرجّع نموذجًا افتراضيًا (stub) حتى تركز على الربط أولاً:
    content = json.dumps(
        {
            "is_correct": False,
            "score": 0.5,
            "error_type": "small_algebra_mistake",
            "short_verdict_ar": "حلّك قريب من الصحيح لكن فيه خطأ جبري بسيط.",
            "main_error_ar": "يبدو أنك أخطأت في إشارة إحدى الحدود أو في توزيع التربيع على القوس.",
            "step_feedback": [],
            "suggested_next_question_ar": "حاول حل المسألة: جد معادلة القطع المكافئ إذا كانت بؤرته (4,0) وخطه الدليل x = -4.",
        },
        ensure_ascii=False,
    )

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # fallback في حال حصل شيء غير متوقع
        return {
            "is_correct": False,
            "score": 0.0,
            "error_type": "unknown",
            "short_verdict_ar": "تعذر تحليل رد النموذج.",
            "main_error_ar": "",
            "step_feedback": [],
            "suggested_next_question_ar": "",
        }


def llm_feedback(
    teacher_eq: str,
    student_eq: str,
    sympy_result: CheckResult,
    question_text: str = "",
    teacher_steps: Optional[List[str]] = None,
    student_steps: Optional[List[str]] = None,
) -> dict:
    sympy_result_dict = {
        "is_correct": sympy_result.is_correct,
        "error_type": sympy_result.error_type,
        "details": sympy_result.details,
        "teacher_expr_str": str(sympy_result.teacher_expr),
        "student_expr_str": str(sympy_result.student_expr),
        "diff_expr_str": str(sympy_result.diff_expr),
    }

    prompt = build_llm_feedback_prompt(
        teacher_eq=teacher_eq,
        student_eq=student_eq,
        sympy_result_dict=sympy_result_dict,
        question_text=question_text,
        teacher_steps=teacher_steps,
        student_steps=student_steps,
    )

    feedback = call_llm(prompt)
    return feedback


# ======================================================
# 3) FastAPI Models
# ======================================================

class GradeRequest(BaseModel):
    # نص السؤال كما استخرجته من OCR
    question_text: Optional[str] = ""
    # المعادلة الصحيحة من المدرس (LaTeX أو نص)
    teacher_equation: str
    # معادلة الطالب (LaTeX أو نص)
    student_equation: str
    # اختياري: خطوات المدرس بصيغة LaTeX أو نصوص
    teacher_steps: Optional[List[str]] = None
    # اختياري: خطوات الطالب إذا كنت تستخرجها
    student_steps: Optional[List[str]] = None


class SympyResultResponse(BaseModel):
    is_correct: bool
    error_type: Optional[str]
    details: Dict[str, Any]
    teacher_expr_str: str
    student_expr_str: str
    diff_expr_str: str


class StepFeedback(BaseModel):
    step_index: int
    is_correct: bool
    comment_ar: str


class LLMFeedbackResponse(BaseModel):
    is_correct: bool
    score: float
    error_type: str
    short_verdict_ar: str
    main_error_ar: str
    step_feedback: List[StepFeedback] = []
    suggested_next_question_ar: str


class GradeResponse(BaseModel):
    sympy_result: SympyResultResponse
    llm_feedback: LLMFeedbackResponse


# ======================================================
# 4) FastAPI app + endpoints
# ======================================================

app = FastAPI(title="Math Grading API", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/grade_equation", response_model=GradeResponse)
def grade_equation(req: GradeRequest):

    # 1) SymPy check
    sym_res = checker.check_equation(req.teacher_equation, req.student_equation)

    sympy_payload = SympyResultResponse(
        is_correct=sym_res.is_correct,
        error_type=sym_res.error_type,
        details=sym_res.details,
        teacher_expr_str=str(sym_res.teacher_expr),
        student_expr_str=str(sym_res.student_expr),
        diff_expr_str=str(sym_res.diff_expr),
    )

    # 2) LLM feedback
    fb = llm_feedback(
        teacher_eq=req.teacher_equation,
        student_eq=req.student_equation,
        sympy_result=sym_res,
        question_text=req.question_text or "",
        teacher_steps=req.teacher_steps,
        student_steps=req.student_steps,
    )

    step_fb_objects = [
        StepFeedback(
            step_index=sf.get("step_index", 0),
            is_correct=sf.get("is_correct", False),
            comment_ar=sf.get("comment_ar", ""),
        )
        for sf in fb.get("step_feedback", [])
    ]

    llm_fb = LLMFeedbackResponse(
        is_correct=fb.get("is_correct", sym_res.is_correct),
        score=float(fb.get("score", 1.0 if sym_res.is_correct else 0.0)),
        error_type=fb.get("error_type", sym_res.error_type or "unknown"),
        short_verdict_ar=fb.get(
            "short_verdict_ar",
            "إجابتك صحيحة." if sym_res.is_correct else "إجابتك غير صحيحة.",
        ),
        main_error_ar=fb.get("main_error_ar", ""),
        step_feedback=step_fb_objects,
        suggested_next_question_ar=fb.get("suggested_next_question_ar", ""),
    )

    return GradeResponse(sympy_result=sympy_payload, llm_feedback=llm_fb)
```

---

## 3️⃣ كيف تشغّله وتختبره؟

### 1. احفظ الملف باسم `main.py`

ثم ثبّت المتطلبات:

```bash
pip install fastapi uvicorn sympy antlr4-python3-runtime
# (و openai أو أي LLM client لو حاب تربطه فعليًا)
```

### 2. شغّل السيرفر:

```bash
uvicorn main:app --reload
```

### 3. جرّب من Postman / curl / متصفح (Swagger)

افتح:
`http://localhost:8000/docs`

وجرب `POST /grade_equation` مثلاً بالـ JSON:

```json
{
  "question_text": "س: جد معادلة القطع المكافئ بطريقة التعريف إذا كانت بؤرته (6,0) وخطه الدليل x = -6.",
  "teacher_equation": "(x - 6)^2 + y^2 = (x + 6)^2",
  "student_equation": "(x - 6)^2 + y^2 = (x + 5)^2",
  "teacher_steps": [
    "MF = MQ",
    "(x - 6)^2 + y^2 = (x + 6)^2",
    "..."
  ],
  "student_steps": [
    "(x - 6)^2 + y^2 = (x + 5)^2"
  ]
}
```

ستحصل على JSON فيه:

* `sympy_result` → تحليل رياضي دقيق.
* `llm_feedback` → (حالياً mock، وبعد ربطه بLLM سيكون حقيقي).

---

## 4️⃣ كيف تربطه مع الـ OCR اللي عملناه؟

* من الـ OCR Pipeline (PRO–Structure v5) عندك:

  * `question_text`
  * معادلة نموذجية (تختار آخر معادلة في الحل مثلًا).
* عندما يرسل الطالب ورقته:

  * تعمل له OCR → معادلة الطالب بصيغة LaTeX أو نص.
* ثم تستدعي backend:

```python
import requests

payload = {
  "question_text": question_text_from_ocr,
  "teacher_equation": teacher_eq_from_ocr,
  "student_equation": student_eq_from_student_ocr
}

r = requests.post("http://your-server:8000/grade_equation", json=payload)
print(r.json())
```

---

لو حاب في خطوة قادمة:

* أضبط لك **تصميم قاعدة بيانات** حول هذا الـ API (جداول الطلاب، الأسئلة، المحاولات، الscores…)
* أو أضيف Endpoint آخر مثل `/grade_full_question` يقرأ JSON السؤال كامل من OCR بدل تمرير معادلة واحدة فقط.
