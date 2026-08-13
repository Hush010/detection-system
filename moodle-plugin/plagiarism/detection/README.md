# AI Content Detection - Moodle Plugin

A Moodle plagiarism detection plugin that integrates with the AI Content Detection system to identify AI-generated content in student assignments.

## Features

- ✅ Detects AI-generated content in text submissions
- ✅ Supports uploaded files (PDF, DOCX, TXT)
- ✅ Shows risk score to instructors before publishing grades
- ✅ Configurable risk threshold
- ✅ Integrates with Moodle 4.1+

## Installation

### Prerequisites
- Moodle 4.1 or higher
- AI Detection API running (e.g., https://detection-system-h6ee.onrender.com)

### Steps

1. **Download or clone** the plugin:
   ```bash
   cd /path/to/moodle/plagiarism
   git clone https://github.com/Hush010/detection-system.git detection
   # OR copy the moodle-plugin/plagiarism/detection folder here
   ```

2. **Visit** `http://yourmoodle.com/admin/index.php` to trigger installation

3. **Configure** at `Site administration > Plugins > Plagiarism > AI Content Detection`:
   - Enable the plugin
   - Set API URL (default: https://detection-system-h6ee.onrender.com)
   - Set risk threshold (0-100)

4. **Test** by submitting an assignment with text content

## Configuration

### API URL
- Default: `https://detection-system-h6ee.onrender.com`
- Change if you self-host the detector

### Risk Threshold
- **0-33**: Low risk
- **34-66**: Medium risk
- **67-100**: High risk

Adjust threshold based on your institution's policy.

## How It Works

1. Student submits an assignment (text or file)
2. Plugin extracts text from submission
3. Sends text to Detection API
4. Stores and displays result to instructor
5. Instructor sees score before publishing grades

## Supported File Types

- ✅ Plain text (.txt)
- ⏳ PDF (.pdf) - In development
- ⏳ Word (.docx) - In development

## Database

The plugin creates one table:
- `mdl_plagiarism_detection_results` - Stores detection scores and labels

## API Integration

The plugin calls the Detection API at:
- **Endpoint**: `/api/analyze`
- **Method**: POST
- **Content-Type**: application/json
- **Payload**: `{"text": "..."}`
- **Response**: `{"result": {"score": 0-100, "label": "...", "explanation": "..."}}`

## Troubleshooting

### Plugin not detecting submissions
- Verify plugin is enabled in admin settings
- Check API URL is accessible
- Review Moodle error logs at `Site administration > Server > Logs`

### Slow detection
- Detection runs asynchronously
- Check Render service status if using the default API

### API connection errors
- Verify firewall allows outbound HTTPS connections
- Test API manually: `curl -X POST https://detection-system-h6ee.onrender.com/api/analyze -H "Content-Type: application/json" -d '{"text": "test"}'`

## License

GNU General Public License v3 or later - See COPYING

## Support

Report issues at: https://github.com/Hush010/detection-system/issues
