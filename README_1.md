ممتاز — هذا هو الانتقال الحقيقي من “تصحيح معادلة” إلى **تصحيح سؤال كامل خطوة بخطوة** ✅
سأضيف لك الآن Endpoint جديد:

> ✅ **POST `/grade_full_question`**
> يأخذ **JSON كامل من مخرجات OCR (PRO–Structure v5)**

* **إجابات الطالب**
  ويعيد:

- تصحيح كل خطوة
- Score لكل خطوة
- Score نهائي للسؤال
- Feedback عربي متكامل

سأعطيك **كود جاهز يندمج مباشرة داخل `main.py` الذي بنيته لك**.

---

# ✅ 1️⃣ شكل Request الجديد: Full Question

هذا هو الشكل الذي سيرسله الـ Frontend بعد OCR:

```json
{
  "question": {
    "question_text": "س : جد معادلة القطع المكافئ بطريقة التعريف...",
    "equation_item_ids": [10],
    "solution": {
      "steps": [
        {
          "index": 0,
          "equation_latex": "(x - 6)^2 + y^2 = (x + 6)^2"
        },
        {
          "index": 1,
          "equation_latex": "x^2 - 12x + 36 + y^2 = x^2 + 12x + 36"
        }
      ]
    }
  },
  "student_answers": {
    "final_equation": "(x - 6)^2 + y^2 = (x + 5)^2",
    "steps": [
      {
        "index": 0,
        "equation_latex": "(x - 6)^2 + y^2 = (x + 5)^2"
      }
    ]
  }
}
```

---

# ✅ 2️⃣ أضف هذه الـ Pydantic Models داخل `main.py`

```python
class SolutionStep(BaseModel):
    index: int
    equation_latex: str

class TeacherSolution(BaseModel):
    steps: List[SolutionStep]

class OCRQuestion(BaseModel):
    question_text: str
    equation_item_ids: List[int]
    solution: TeacherSolution

class StudentAnswer(BaseModel):
    final_equation: str
    steps: Optional[List[SolutionStep]] = []

class FullGradeRequest(BaseModel):
    question: OCRQuestion
    student_answers: StudentAnswer


class StepGrade(BaseModel):
    step_index: int
    sympy_correct: bool
    sympy_error_type: Optional[str]
    llm_feedback: LLMFeedbackResponse


class FullGradeResponse(BaseModel):
    final_score: float
    final_verdict_ar: str
    steps_result: List[StepGrade]
```

---

# ✅ 3️⃣ أضف هذا Endpoint أسفل `/grade_equation`

```python
@app.post("/grade_full_question", response_model=FullGradeResponse)
def grade_full_question(req: FullGradeRequest):

    teacher_steps = req.question.solution.steps
    student_steps = req.student_answers.steps or []

    steps_result = []
    total_score = 0.0
    counted_steps = 0

    # نص السؤال
    question_text = req.question.question_text

    # === تصحيح كل خطوة ===
    for t_step in teacher_steps:

        # نحاول نطابق خطوة الطالب بنفس index
        s_step = next(
            (s for s in student_steps if s.index == t_step.index),
            None
        )

        if s_step is None:
            # الطالب لم يكتب هذه الخطوة
            sym_res = checker.check_equation(
                t_step.equation_latex, "0"
            )

            fb = llm_feedback(
                teacher_eq=t_step.equation_latex,
                student_eq="",
                sympy_result=sym_res,
                question_text=question_text
            )

            step_score = fb.get("score", 0.0)

        else:
            sym_res = checker.check_equation(
                t_step.equation_latex,
                s_step.equation_latex
            )

            fb = llm_feedback(
                teacher_eq=t_step.equation_latex,
                student_eq=s_step.equation_latex,
                sympy_result=sym_res,
                question_text=question_text
            )

            step_score = fb.get("score", 0.0)

        total_score += step_score
        counted_steps += 1

        step_feedback_obj = StepGrade(
            step_index=t_step.index,
            sympy_correct=sym_res.is_correct,
            sympy_error_type=sym_res.error_type,
            llm_feedback=LLMFeedbackResponse(
                is_correct=fb.get("is_correct", False),
                score=float(fb.get("score", 0.0)),
                error_type=fb.get("error_type", "unknown"),
                short_verdict_ar=fb.get("short_verdict_ar", ""),
                main_error_ar=fb.get("main_error_ar", ""),
                step_feedback=[
                    StepFeedback(
                        step_index=x.get("step_index", 0),
                        is_correct=x.get("is_correct", False),
                        comment_ar=x.get("comment_ar", "")
                    )
                    for x in fb.get("step_feedback", [])
                ],
                suggested_next_question_ar=fb.get("suggested_next_question_ar", "")
            )
        )

        steps_result.append(step_feedback_obj)

    # === حساب النتيجة النهائية ===
    final_score = total_score / max(1, counted_steps)

    if final_score > 0.85:
        verdict = "إجابة ممتازة ✅"
    elif final_score > 0.6:
        verdict = "إجابة جيدة مع بعض الأخطاء ⚠️"
    elif final_score > 0.3:
        verdict = "إجابة ضعيفة وتحتاج مراجعة ❌"
    else:
        verdict = "الإجابة غير صحيحة تقريبًا ❌"

    return FullGradeResponse(
        final_score=round(final_score, 2),
        final_verdict_ar=verdict,
        steps_result=steps_result
    )
```

---

# ✅ 4️⃣ كيف تختبره في Swagger؟

بعد تشغيل:

```bash
uvicorn main:app --reload
```

افتح:

```
http://localhost:8000/docs
```

اختر:

> ✅ `POST /grade_full_question`

وألصق Request مثل هذا:

```json
{
  "question": {
    "question_text": "س : جد معادلة القطع المكافئ بطريقة التعريف...",
    "equation_item_ids": [10],
    "solution": {
      "steps": [
        {
          "index": 0,
          "equation_latex": "(x - 6)^2 + y^2 = (x + 6)^2"
        },
        {
          "index": 1,
          "equation_latex": "x^2 - 12x + 36 + y^2 = x^2 + 12x + 36"
        }
      ]
    }
  },
  "student_answers": {
    "final_equation": "(x - 6)^2 + y^2 = (x + 5)^2",
    "steps": [
      {
        "index": 0,
        "equation_latex": "(x - 6)^2 + y^2 = (x + 5)^2"
      }
    ]
  }
}
```

ستحصل على Response مثل:

```json
{
  "final_score": 0.5,
  "final_verdict_ar": "إجابة ضعيفة وتحتاج مراجعة ❌",
  "steps_result": [
    {
      "step_index": 0,
      "sympy_correct": false,
      "sympy_error_type": "coefficient_mismatch",
      "llm_feedback": {
        "is_correct": false,
        "score": 0.5,
        "error_type": "small_algebra_mistake",
        "short_verdict_ar": "حلّك قريب من الصحيح.",
        "main_error_ar": "خطأ في قيمة الإشارة داخل التربيع.",
        "step_feedback": [],
        "suggested_next_question_ar": "حاول مسألة مشابهة..."
      }
    }
  ]
}
```

---

# ✅ الآن أصبح لديك Backend حقيقي يدعم:

| الميزة                   | الحالة |
| ------------------------ | ------ |
| OCR استخراج سؤال كامل    | ✅      |
| تصحيح معادلة واحدة       | ✅      |
| تصحيح حل كامل خطوة بخطوة | ✅      |
| Hybrid (SymPy + LLM)     | ✅      |
| API جاهز للواجهة         | ✅      |

---

# 🚀 الخطوة التالية المقترحة (جاهز متى ما قلت):

1. ✅ إضافة **Student Attempts History**
2. ✅ إضافة **Database Schema**
3. ✅ إضافة **User Accounts**
4. ✅ إضافة **Session Tutor Engine**
5. ✅ إضافة **Auto Generated Homework**

---

## 🔥 سؤالي لك الآن:

هل تريد أن أبني لك مباشرة:

> ✅ **تصميم قاعدة بيانات (PostgreSQL أو SQLite) لكل هذا النظام؟**
