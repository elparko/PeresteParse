# Pereste Parse Distribution Plan

## Overview
Host the unsigned Pereste Parse.dmg (263MB) for public download via GitHub Releases and create a landing page with installation instructions.

## Implementation Steps

### 1. Prepare GitHub Repository
- Make repository public (if not already)
- Verify .gitignore excludes large files (models, dist/, build/)
- Ensure README.md has basic project info

### 2. Create GitHub Release
- Tag version (e.g., v1.0.0)
- Upload DMG file as release asset
- Write release notes describing features
- Use `gh release create` command

### 3. Create Landing Page
Two options:
- **Option A**: Use GitHub Pages with custom domain
- **Option B**: Use existing domain with static hosting

Landing page should include:
- App description and features
- Download button linking to latest release DMG
- Installation instructions (see below)
- System requirements (macOS 11.0+, Apple Silicon or Intel)
- Screenshots (optional)

### 4. Installation Instructions for Users
Critical: Include clear instructions for bypassing macOS Gatekeeper warning:

```
## Installation

1. Download Pereste Parse.dmg
2. Open the DMG and drag Pereste Parse to Applications
3. **First Launch**: Right-click (or Control+click) on Pereste Parse and select "Open"
4. Click "Open" in the security dialog
5. The app will now run normally on subsequent launches

**Note**: This app is not code-signed. macOS will show a security warning on first launch.
The right-click method bypasses this for trusted applications.
```

### 5. Optional Enhancements
- Add GitHub badge showing latest release version
- Create a simple website with Tailwind/minimal CSS
- Add analytics to track downloads
- Include FAQ section about the security warning

## Commands to Execute

```bash
# 1. Create and push a git tag
git tag -a v1.0.0 -m "Initial release of Pereste Parse"
git push origin v1.0.0

# 2. Create GitHub release with DMG
gh release create v1.0.0 \
  "dist/Pereste Parse.dmg" \
  --title "Pereste Parse v1.0.0" \
  --notes "First public release. Features: multi-model support, CSV import/export, queue system, HTML notes formatting."

# 3. Enable GitHub Pages (if using)
# Go to Settings > Pages > Source: gh-pages branch or docs/ folder
```

## File Structure

```
pereste-parse/
├── dist/
│   └── Pereste Parse.dmg (upload to GitHub Releases)
├── docs/                  (or separate gh-pages branch)
│   ├── index.html        (landing page)
│   └── screenshot.png    (optional)
├── README.md
└── DISTRIBUTION_PLAN.md
```

## Security Considerations

Since the app is **not code-signed**:
- Users will see "unverified developer" warning
- Cannot be distributed via Mac App Store
- Right-click > Open workaround required on first launch
- Some corporate/educational networks may block unsigned apps

## Success Criteria

- [ ] DMG uploaded to GitHub Releases
- [ ] Landing page live with download link
- [ ] Clear installation instructions visible
- [ ] README updated with project info
- [ ] Repository is public
- [ ] Download link tested and works

## Timeline

- **Step 1-2**: 10 minutes (repository prep, create release)
- **Step 3-4**: 30-60 minutes (create landing page, write docs)
- **Step 5**: Optional, time varies

Total: ~1 hour for full deployment
