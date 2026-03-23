# Image Classifier

Image Classifier is a polished browser-based vision demo that analyzes uploaded images directly on the client side. It estimates a best-fit category, surfaces confidence scores, breaks down visual signals like brightness and contrast, and stores recent analyses in local history.

## Live Demo

- Portfolio route: https://rabin-portfolio-rust.vercel.app/image-classifier/
- Portfolio homepage: https://rabin-portfolio-rust.vercel.app/
- Repository: https://github.com/rabinregmi188/Image-Classifier

## Highlights

- Drag-and-drop upload flow with instant preview
- Client-side image analysis using the Canvas API
- Top-3 prediction scoring with animated confidence bars
- Signal breakdown cards for brightness, contrast, saturation, and scene bias
- Local history saved with `localStorage`
- Responsive dashboard with colorful glassmorphism styling

## Stack

- JavaScript
- HTML
- CSS
- Canvas API
- LocalStorage

## Run locally

Open `public/index.html` in a browser, or serve the folder with a simple static server.

## Project Notes

This version is intentionally static so it can deploy cleanly inside the portfolio, the same way the other live project routes work.
