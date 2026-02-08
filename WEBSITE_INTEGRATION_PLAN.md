# Website Integration Plan for elparko.com

## Overview
Integrate Pereste Parse download and information into existing elparko.com website instead of using GitHub Pages.

## Assets Needed
- DMG download link: `https://github.com/elparko/PeresteParse/releases/latest/download/Pereste%20Parse.dmg`
- App icon/logo: `static/plogo-icon.png` and `static/Plogo-large.png`
- Screenshots (if desired - can be added later)

## Content to Add

### Product Page Section
Create a section or page on elparko.com with:

**Hero Section:**
- Title: "Pereste Parse - Privacy-First Study App"
- Subtitle: "Flashcard study app powered by local LLMs for macOS"
- Download button linking to GitHub release DMG

**Features Grid:**
- 🤖 Multi-Model Support - Choose from 4 optimized LLM models
- 🎤 Voice Input - Audio transcription for hands-free notes
- 🔒 Privacy First - All data stored locally on your Mac
- ⚡ Fast Queue - Input multiple entries while parsing
- 📝 Auto Flashcards - AI-powered flashcard generation
- 💾 CSV Import/Export - Backup and migrate your data

**Installation Instructions:**
```
1. Download Pereste Parse.dmg
2. Open the DMG file
3. Drag Pereste Parse to Applications
4. First Launch: Right-click → Open → Click "Open" in dialog
5. App will run normally on future launches
```

**Security Note Box:**
> ⚠️ This app is not code-signed. macOS will show an "unverified developer" warning on first launch. Use the right-click method above to bypass this for trusted applications.

**System Requirements:**
- macOS 11.0+ (Big Sur or later)
- Apple Silicon (M1/M2/M3) or Intel
- 4GB+ RAM recommended
- 5GB free disk space

**Links:**
- Download: [GitHub Release](https://github.com/elparko/PeresteParse/releases/latest)
- Documentation: [README](https://github.com/elparko/PeresteParse)
- Report Issues: [GitHub Issues](https://github.com/elparko/PeresteParse/issues)
- Security: [SECURITY.md](https://github.com/elparko/PeresteParse/blob/main/SECURITY.md)

## Technical Details

**File Size:** ~263MB
**Version:** v2.0.0
**License:** MIT
**Platform:** macOS only

## Optional Enhancements

1. **Screenshots/Demo Video**
   - Main interface screenshot
   - Settings/configuration screen
   - Queue system in action
   - Model download interface

2. **Use Cases Section**
   - Medical students studying anatomy
   - Language learners creating vocabulary cards
   - Professional certification prep
   - General knowledge retention

3. **FAQ Section**
   - Why is it so large? (Includes 4 LLM models)
   - Is my data private? (Yes, everything local)
   - Do I need internet? (Only for cloud API option)
   - Can I use my own models? (Yes, HuggingFace integration)

4. **Analytics** (Optional)
   - Track download button clicks
   - Track which links users click (GitHub vs direct download)

## Future Considerations

- Consider code-signing if app gains traction (requires Apple Developer account $99/year)
- Add update notification system in future versions
- Consider Sparkle framework for auto-updates
- Add telemetry opt-in for crash reporting (privacy-preserving)

## Integration Notes

- Keep download link pointing to GitHub releases (canonical source)
- Update version number when new releases published
- Consider adding RSS feed or newsletter for update notifications
- Link back to GitHub for open-source transparency
