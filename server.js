const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

// --- Static File Serving ---
app.use(express.static(path.join(__dirname, 'public')));

// --- Routes ---
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// --- Server Start ---
app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
    console.log('The image conversion will now happen entirely on the client-side (in your browser).');
});