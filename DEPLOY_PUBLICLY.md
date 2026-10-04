# Deploying this tool publicly (the RELIABLE way)

The Jupyter + ngrok setup used during development is fine for quick local
testing, but it is NOT reliable for a genuinely public tool: the link
expires when the notebook stops, ngrok needs an auth token, and anyone with
the free ngrok link can see a "visit this site" warning page.

For a real public deployment, use **Streamlit Community Cloud** (free,
built by the same team as Streamlit, designed exactly for this):

## Steps

1. Create a free GitHub account if you don't have one: https://github.com
2. Create a new PUBLIC repository (e.g. `2d-flash-dose`)
3. Upload every file from this folder (`app.py`, `requirements.txt`, the
   `dose2d/` folder, `README.md`) to that repository -- GitHub's web
   interface lets you drag-and-drop files directly, no git command line
   needed if you prefer
4. Go to https://share.streamlit.io and sign in with your GitHub account
   (free)
5. Click "New app", select your repository, set the main file to `app.py`
6. Click Deploy

That's it -- within a few minutes you get a permanent public link like
`https://your-app-name.streamlit.app` that:
- Never expires (as long as the repo exists)
- Needs no ngrok, no auth token, no Jupyter running
- Updates automatically whenever you push changes to GitHub
- Is genuinely shareable with a supervisor, reviewers, or the public

This is the standard, professional way research tools like this get shared
publicly, and is what should be used for anything beyond local testing.
