Byte2Beat Cardiovascular Health Toolkit
=======================================

There are two models that can be generated, in model/ and model-death/.
For convenience, pre-generated .joblib files are already included.

The backend/ directory contains an API server for remote usage of the models.

The frontend/ directory contains a web app that uses the API.

Python environments and packages are managed using uv (https://docs.astral.sh/uv/).

The frontend requires Node.js and npm (https://nodejs.org/en/download).


Basic usage
-----------

1. Install dependencies (uv, Node.js, npm).
2. (optional) Re-create the models according to their README files.
3. Run the backend:

    cd backend
    uv run api.py

4. Run the frontend:

    cd frontend
    npm run dev

5. Open the web app in browser (the URL should be visible in the console).


Data sets citations:

NHANES:                             https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023
NHANES NDI Linked Mortality Files:  https://www.cdc.gov/nchs/linked-data/mortality-files/index.html
