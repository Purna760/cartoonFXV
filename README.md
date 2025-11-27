# CartoonFX - Video to Cartoon Converter

## Overview

CartoonFX is a professional Flask-based web application that converts uploaded video files into cartoon-style videos using advanced computer vision techniques. The application accepts video uploads (MP4, AVI, MOV, MKV formats), applies one of eight different cartoonization styles to each frame using OpenCV, and returns the processed video for download. The interface features a modern, glassmorphism-inspired design with gradient backgrounds and professional styling.

## Recent Changes

**November 26, 2025 - Major UI Overhaul & New Filters**
- Added 2 new professional filters (total 8 styles):
  - **AnimeGAN:** Realistic anime transformation with enhanced saturation and soft edges
  - **Oil Painting:** Classic artistic oil paint effect with smooth color blending
- Reordered filter display: Comic Book → Pop Art → Pencil Sketch → AnimeGAN → Oil Painting → Classic Cartoon → Anime → Watercolor
- Complete UI redesign with:
  - Modern gradient backgrounds and glassmorphism effects
  - Professional "CartoonFX" branding with logo
  - Enhanced upload zone with better styling and format badges
  - Improved 4-column style selection grid (2 on mobile)
  - Animated processing section with spinner
  - Side-by-side video comparison preview
  - Features section showcasing all 8 filter styles
  - Redesigned How It Works section with step cards
  - About section with statistics (8 styles, 500MB, HD, Free)
  - Professional dark footer with branding

**Previous - Multiple Cartoon Styles Feature**
- Original 6 cartoon styles:
  - **Classic Cartoon:** Bold edges with smooth bilateral-filtered colors
  - **Anime:** Soft edges with enhanced saturation and vibrant colors
  - **Comic Book:** Strong black outlines with posterized/quantized colors
  - **Watercolor:** Soft, painterly effect with heavy bilateral filtering
  - **Pop Art:** High contrast posterized colors with bold saturation
  - **Pencil Sketch:** Black and white dodge-burn sketch effect

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework:** Server-rendered HTML with Tailwind CSS and vanilla JavaScript
- **Rationale:** Lightweight approach suitable for utility applications with straightforward interactions
- **Design System:** Material Design with Inter font family, using Tailwind's utility classes for responsive layouts
- **Component Structure:** Single-page application with progressive disclosure - upload zone, processing feedback, and result preview
- **Responsive Strategy:** Mobile-first single column, expanding to two-column preview layout on desktop (lg:grid-cols-2)

### Backend Architecture

**Framework:** Flask (Python)
- **Rationale:** Lightweight web framework suitable for file handling and synchronous video processing
- **File Handling:** Werkzeug utilities for secure filename handling and file upload management
- **Configuration:**
  - Session secret from environment variables
  - ProxyFix middleware for proper header handling behind reverse proxies
  - Maximum file upload size: 500MB
  - Allowed file extensions: mp4, avi, mov, mkv

**Processing Pipeline:**
1. **Upload Validation:** File type and size verification
2. **Video Processing:** Frame-by-frame cartoonization using OpenCV
3. **Cartoonization Algorithm:**
   - Convert frame to grayscale and apply median blur
   - Detect edges using adaptive thresholding
   - Apply bilateral filter for color smoothing
   - Combine edges with smoothed color using bitwise AND
4. **Output Generation:** Re-encode processed frames into output video

**File Organization:**
- `static/uploads/`: Temporary storage for uploaded videos
- `static/processed/`: Storage for processed cartoon videos
- UUID-based filename generation to prevent collisions

### Data Storage Solutions

**Storage Type:** Local filesystem
- **Rationale:** Simple file-based storage suitable for temporary processing workflow
- **Approach:** No database required; files stored directly in static directories
- **Cleanup Strategy:** Not implemented in current codebase (potential improvement area)

### Authentication and Authorization

**Current State:** None implemented
- **Note:** Application appears designed for public use without user accounts or access control
- **Session Management:** Flask session secret configured but not actively used for authentication

### Design System

**Typography Hierarchy:**
- Page Title: text-3xl, font-bold (48px)
- Section Headers: text-xl, font-semibold (30px)
- Body Text: text-base (16px)
- UI Labels: text-sm, font-medium (14px)
- Helper Text: text-xs (12px)

**Spacing System:** Tailwind spacing units (2, 4, 6, 8, 12, 16, 24)
- Component padding: p-6 or p-8
- Section spacing: mb-8 or mb-12
- Container max-width: max-w-6xl

**Interaction Patterns:**
- Drag-and-drop upload zone with hover states
- Visual feedback during processing (progress indicators, spinning animations)
- Responsive border colors and transforms on interaction

## External Dependencies

### Python Libraries

**OpenCV (cv2):** Computer vision and video processing
- **Purpose:** Frame extraction, image manipulation, video encoding/decoding
- **Key Functions:** Video capture, cartoon effect filters (bilateral filter, adaptive thresholding, bitwise operations)

**Flask:** Web framework
- **Purpose:** HTTP request handling, routing, template rendering
- **Extensions:** 
  - Werkzeug utilities (secure_filename, ProxyFix)
  - Built-in file upload handling

**NumPy:** Numerical operations
- **Purpose:** Array manipulation for image data processing
- **Usage:** Implicit dependency for OpenCV operations

### Frontend Dependencies

**Tailwind CSS:** Utility-first CSS framework (via CDN)
- **Purpose:** Responsive design, component styling
- **Delivery:** CDN-based (cdn.tailwindcss.com)

**Google Fonts:** Typography
- **Font Family:** Inter (weights: 400, 500, 600, 700)
- **Delivery:** CDN-based

### Environment Variables

**SESSION_SECRET:** Flask session encryption key
- **Required:** Yes (though not actively used for authentication)
- **Purpose:** Session data security

### File Format Support

**Input Formats:** MP4, AVI, MOV, MKV
**Output Format:** Determined by OpenCV VideoWriter codec configuration (implementation incomplete in provided code)
