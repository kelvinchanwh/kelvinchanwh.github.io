# Copilot Instructions for Jekyll Personal Website

## Project Overview
This is a Jekyll-based personal website project. It uses Jekyll 4.3.1 for static site generation, with a single-page layout design focused on portfolio/projects display. The site is deployed to GitHub Pages.

For detailed setup and customization, see [README.md](../README.md).

## Build and Run
- **Development**: `npm run dev` - Starts local Jekyll server
- **Production Build**: `npm run predeploy` - Builds to `_site/` with production environment
- **Deploy**: `npm run deploy` - Pushes `_site/` to gh-pages branch

Note: Uses Bundler for Ruby gems; ensure Ruby and Bundler are installed.

## Architecture
- **Layout**: Single template in `_layouts/default.html` handling navbar, footer, and SEO
- **Styles**: Sass-based with variables in `_sass/vars.scss`; main file `css/main.scss`
- **Content**: Markdown files in root or `projects/` with YAML front matter
- **Assets**: Images in `images/` and `projects/images/`
- **Output**: Generated site in `_site/`

## Conventions
- Content files: Markdown (.md) with `layout: default`
- Navigation: Configured in `_config.yml` under `nav` array
- Page titles: Extracted from H1 headings
- Images: Relative paths; use specific classes for styling
- Colors: Customize `$dark-accent`, `$light-accent` in `_sass/vars.scss`

## Potential Pitfalls
- `favicon_location` referenced but not defined in config; may need custom handling
- `baseurl` is commented out; links use `site.url` + `site.baseurl`
- Sass output is compressed; no source maps for debugging
- Local Jekyll version may differ from GitHub Pages; use `github-pages` gem for consistency if needed
- No post collections; not designed for blogging

## Key Files
- [_config.yml](../_config.yml): Site configuration, navigation, plugins
- [_layouts/default.html](../_layouts/default.html): Main template
- [_sass/vars.scss](../_sass/vars.scss): Theme variables
- [index.md](../index.md): Home page
- [projects/airbnb-analysis.md](../projects/airbnb-analysis.md): Example project page

Avoid duplicating content from README.md; refer to it for theme-specific instructions.