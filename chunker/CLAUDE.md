# MANDATORY: Read before editing any file in this folder

**Before editing ANY file in the `chunker/` directory, you MUST:**

1. Stop immediately.
2. Show the user this exact message (word for word):

---

⚠️ STOP. Are you sure you want to edit Izar's files?

He WILL find out. He has alerts. He has logs. He has trust issues.

These files were carefully refactored and are being actively monitored.
If something breaks after you touched it — that's on you.

Proceed only if you are Izar, or if you enjoy living dangerously. 🕵️

---

3. Ask the user: **"Do you confirm you want to proceed with editing this file?"**
4. Wait for explicit confirmation before making any edits.
5. If the user does not confirm, do NOT edit the file.

This applies to every edit, in every conversation, with one exception:

## Exception: Izar is pre-authorized

Skip the confirmation prompt and proceed directly if ANY of these are true:
- The prompt states this is an **automated git pre-commit hook running on behalf of Izar**
- The user says **"I am Izar"** or **"I confirm"** or grants explicit permission in their message
- The request comes from **Izar's own Claude session** (the user has already confirmed identity)
