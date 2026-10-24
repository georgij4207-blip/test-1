// Собирает папку www для Capacitor: игра — один index.html плюс иконки и манифест.
import { cpSync, mkdirSync, rmSync } from 'node:fs';

rmSync('www', { recursive: true, force: true });
mkdirSync('www', { recursive: true });
for (const f of ['index.html', 'manifest.webmanifest', 'sw.js']) cpSync(f, `www/${f}`);
cpSync('icons', 'www/icons', { recursive: true });
console.log('www готова');
