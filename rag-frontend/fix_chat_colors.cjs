const fs = require('fs');
const path = require('path');

const filepath = path.join(__dirname, 'src', 'components', 'chat', 'chatWindow.jsx');
let content = fs.readFileSync(filepath, 'utf8');

const replacements = {
    '"#faf9f5"': '"var(--app-bg)"',
    '"rgba(250,249,245,0.94)"': '"var(--app-surface)"',
    '"#e8e2d6"': '"var(--app-border)"',
    '"#e3ded2"': '"var(--app-border)"',
    '"#e8e4da"': '"var(--app-surface-muted)"',
    '"#4d4942"': '"var(--app-text)"',
    '"#2f2b25"': '"var(--app-text)"',
    '"#d8d1c3"': '"var(--app-border-strong)"',
    '"#050505"': '"var(--app-surface-muted)"',
    '"rgba(255,255,255,0.16)"': '"var(--app-border-strong)"'
};

for (const [oldVal, newVal] of Object.entries(replacements)) {
    content = content.split(oldVal).join(newVal);
}

fs.writeFileSync(filepath, content, 'utf8');
console.log('Successfully replaced hardcoded colors in ChatWindow.jsx');
