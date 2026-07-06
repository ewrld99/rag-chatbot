const fs = require('fs');
const path = require('path');

const directory = path.join(__dirname, 'src');

const replacements = {
    "#f0eee7": "var(--app-bg)",
    "#fffdf8": "var(--app-surface)",
    "#fffaf0": "var(--app-surface-muted)",
    "#f7f2e8": "var(--app-panel)",
    "#ded9cd": "var(--app-border)",
    "#d7d0c1": "var(--app-border-strong)",
    "#2b2925": "var(--app-text)",
    "#6f6a61": "var(--app-muted)",
    "#8a8478": "var(--app-faint)",
    "#9a4f35": "var(--app-accent)",
    "#d96c47": "var(--app-accent-strong)",
    "#f7eadf": "var(--app-accent-soft)",
    "#a13f24": "var(--app-danger)",
    "#fff0e8": "var(--app-danger-soft)"
};

function walkSync(currentDirPath, callback) {
    fs.readdirSync(currentDirPath).forEach(function (name) {
        var filePath = path.join(currentDirPath, name);
        var stat = fs.statSync(filePath);
        if (stat.isFile()) {
            callback(filePath, stat);
        } else if (stat.isDirectory()) {
            walkSync(filePath, callback);
        }
    });
}

walkSync(directory, function (filePath) {
    if (filePath.endsWith('.jsx') || filePath.endsWith('.css') || filePath.endsWith('.js')) {
        if (path.basename(filePath) === 'index.css') {
            return;
        }

        let content = fs.readFileSync(filePath, 'utf8');
        let modified = false;

        // Create a regex to match hex colors case-insensitively, 
        // but only if they are not already inside var(--...)
        for (const [hex, cssVar] of Object.entries(replacements)) {
            const regex = new RegExp(hex, 'gi');
            if (regex.test(content)) {
                content = content.replace(regex, cssVar);
                modified = true;
            }
        }

        if (modified) {
            fs.writeFileSync(filePath, content, 'utf8');
            console.log(`Updated ${filePath}`);
        }
    }
});

console.log("Theme migration complete!");
