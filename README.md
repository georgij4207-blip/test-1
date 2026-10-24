# test-1

## Искорка

Игра — один файл `index.html` (открывается в браузере, ставится как PWA).

### Android

Приложение собирается через [Capacitor](https://capacitorjs.com): `npm ci && npm run sync` копирует игру в `www/` и в проект `android/`.

При каждом изменении GitHub Actions (`.github/workflows/android.yml`) собирает отладочный APK. Скачать его можно на вкладке **Actions** → нужный запуск → **Artifacts** → `iskorka-debug-apk`. Отладочные сборки подписаны постоянным ключом `android/app/debug.keystore`, поэтому новая версия ставится поверх старой.
