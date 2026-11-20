# Contributing to AeroPulse

Thank you for your interest in contributing to AeroPulse! We welcome contributions from ML engineers, aerospace practitioners, and frontend developers.

## Code of Conduct
Please maintain a respectful, constructive, and evidence-grounded discussion environment.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shaaunak/aeropulse.git
   cd aeropulse
   ```

2. **Python Environment (Python 3.10+, tested on 3.12)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Frontend Environment (Node 18+)**:
   ```bash
   cd frontend
   npm install
   ```

## Running Tests & Checks

Before submitting a Pull Request, ensure all checks pass:

- **Backend tests**:
  ```bash
  make test
  ```
- **Frontend typecheck & build**:
  ```bash
  make build
  ```
- **Benchmark reproducibility**:
  ```bash
  make benchmark
  ```

## Scientific Rigor Rules
- **No data leakage**: Keep train and validation sets engine-disjoint. Fit scalers and feature selection only on training engines.
- **Reproducibility**: Always document random seeds and report mean ± standard deviation across multiple seeds when proposing model modifications.
- **Official evaluation**: Do not alter or cherry-pick the official 100 test engine labels.
