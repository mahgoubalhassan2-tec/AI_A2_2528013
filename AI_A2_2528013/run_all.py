
from pathlib import Path
import argparse
import subprocess
import sys

# تحديد المجلد الرئيسي للمشروع
ROOT = Path(__file__).resolve().parent


def main():
    # استقبال رقم التسجيل والبذرة من سطر الأوامر
    parser = argparse.ArgumentParser(
        description="Run the complete AI assignment pipeline"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Your student registration number"
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The last four digits of your student ID"
    )

    args = parser.parse_args()

    # استخراج الأرقام فقط من رقم التسجيل
    digits = "".join(
        character
        for character in args.student_id
        if character.isdigit()
    )

    # التحقق من أن رقم التسجيل يحتوي على أربعة أرقام على الأقل
    if len(digits) < 4:
        parser.error("Student ID must contain at least 4 digits.")

    # استخراج آخر أربعة أرقام
    expected_seed = int(digits[-4:])

    # التأكد من تطابق البذرة مع رقم التسجيل
    if args.seed != expected_seed:
        parser.error(
            f"Incorrect seed. Expected: {expected_seed}"
        )

    # ملفات المشروع التي يجب تشغيلها بالترتيب
    scripts = [
        "data/generate_data.py",
        "src/clean_data.py",
        "src/validate_data.py",
        "src/train_models.py",
        "src/train_classification.py",
        "src/train_clustering.py",
        "src/build_artifacts.py",
        "src/create_documents.py",
    ]

    # تشغيل كل مرحلة من مراحل المشروع
    for script in scripts:
        script_path = ROOT / script

        # التأكد من وجود الملف قبل تشغيله
        if not script_path.exists():
            print(f"ERROR: File not found: {script_path}")
            sys.exit(1)

        print("\n" + "=" * 50)
        print(f"Running: {script}")
        print("=" * 50)

        # تجهيز أمر تشغيل الملف
        command = [
            sys.executable,
            str(script_path)
        ]

        # تمرير رقم التسجيل والبذرة إلى مولّد البيانات
        if script == "data/generate_data.py":
            command.extend([
                "--student-id",
                args.student_id,
                "--seed",
                str(args.seed)
            ])

        # تشغيل المرحلة وإيقاف البرنامج إذا فشلت
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True
            )
        except subprocess.CalledProcessError as error:
            print(f"\nERROR: {script} failed.")
            print(f"Exit code: {error.returncode}")
            sys.exit(error.returncode)

    # رسالة النجاح بعد اكتمال جميع المراحل
    print("\n" + "=" * 50)
    print("Verification code: AI-A2-8013")
    print("All pipeline stages completed successfully!")
    print("=" * 50)


# بدء البرنامج عند تشغيل هذا الملف مباشرة
if __name__ == "__main__":
    main()
