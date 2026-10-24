# Contributing to ContextPilot

Thank you for your interest in contributing to **ContextPilot**! We welcome bug reports, feature enhancements, documentation improvements, and evaluation benchmarks.

---

## Code of Conduct
Please ensure a respectful, inclusive, and professional environment for all contributors.

---

## Development Workflow

1. **Fork the Repository**:
   Click the "Fork" button on GitHub and clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/ContextPilot.git
   cd ContextPilot
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Tests & Benchmark**:
   ```bash
   python scripts/run_eval.py
   python scripts/run_experiments.py
   ```

5. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```

6. **Submit a Pull Request**:
   - Ensure all code is cleanly formatted and typed.
   - Describe the purpose and scope of your changes in the PR description.
