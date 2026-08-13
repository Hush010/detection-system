<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Main plugin class for AI Content Detection
 *
 * @package    plagiarism_detection
 * @copyright  2024
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

require_once $CFG->libdir . '/filelib.php';
require_once $CFG->dirroot . '/plagiarism/lib.php';

class plagiarism_plugin_detection extends plagiarism_plugin {

    public function get_configs() {
        return array(
            'detection_api_url',
            'detection_enabled',
            'detection_threshold'
        );
    }

    public function hook_assess_submission($linkarray) {
        global $DB, $OUTPUT;

        // Get submission data
        $submissionid = $linkarray['submissionid'];
        $cmid = $linkarray['cmid'];

        if (!$this->is_plugin_configured()) {
            return '';
        }

        // Check if already analyzed
        $result = $DB->get_record(
            'plagiarism_detection_results',
            array('submissionid' => $submissionid)
        );

        if ($result) {
            return $this->render_result($result);
        }

        return '';
    }

    public function hook_file_uploaded($eventdata) {
        $this->check_submission_files($eventdata);
    }

    public function hook_assessable_uploaded($eventdata) {
        $this->check_submission_files($eventdata);
    }

    public function hook_assessable_text_uploaded($eventdata) {
        $this->check_submission_text($eventdata);
    }

    private function check_submission_files($eventdata) {
        global $DB;

        $submissionid = $eventdata->submission->id ?? null;
        if (!$submissionid) {
            return;
        }

        $fs = get_file_storage();
        $files = $fs->get_area_files(
            $eventdata->context->id,
            'assignsubmission_file',
            'submission_files',
            $eventdata->submission->id
        );

        foreach ($files as $file) {
            if ($file->is_directory()) {
                continue;
            }

            // Extract text from file
            $text = $this->extract_text_from_file($file);
            if ($text) {
                $this->analyze_text($text, $submissionid);
            }
        }
    }

    private function check_submission_text($eventdata) {
        global $DB;

        $submissionid = $eventdata->submission->id ?? null;
        $text = $eventdata->submission->submission_text ?? null;

        if ($submissionid && $text) {
            $this->analyze_text($text, $submissionid);
        }
    }

    private function extract_text_from_file($file) {
        global $CFG;

        $filename = $file->get_filename();
        $mimetype = $file->get_mimetype();

        // Handle different file types
        if ($mimetype === 'text/plain') {
            return $file->get_content();
        } elseif ($mimetype === 'application/pdf') {
            return $this->extract_text_from_pdf($file);
        } elseif ($mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
            return $this->extract_text_from_docx($file);
        }

        return null;
    }

    private function extract_text_from_pdf($file) {
        // For now, just return null - can implement PDF parsing later
        // or use pdfparser library
        return null;
    }

    private function extract_text_from_docx($file) {
        // For now, just return null - can implement DOCX parsing later
        return null;
    }

    private function analyze_text($text, $submissionid) {
        global $DB, $CFG;

        $api_url = get_config('plagiarism_detection', 'detection_api_url');
        if (!$api_url) {
            return;
        }

        // Call detection API
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $api_url . '/api/analyze');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode(array('text' => $text)));
        curl_setopt($ch, CURLOPT_TIMEOUT, 30);

        $response = curl_exec($ch);
        $httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpcode === 200) {
            $data = json_decode($response, true);
            if (isset($data['result'])) {
                $result = $data['result'];
                $this->store_result($submissionid, $result);
            }
        }
    }

    private function store_result($submissionid, $result) {
        global $DB;

        $record = new stdClass();
        $record->submissionid = $submissionid;
        $record->score = $result['score'] ?? 0;
        $record->label = $result['label'] ?? 'Unknown';
        $record->timestamp = time();

        $existing = $DB->get_record(
            'plagiarism_detection_results',
            array('submissionid' => $submissionid)
        );

        if ($existing) {
            $record->id = $existing->id;
            $DB->update_record('plagiarism_detection_results', $record);
        } else {
            $DB->insert_record('plagiarism_detection_results', $record);
        }
    }

    private function render_result($result) {
        $label_class = 'label-success';
        if ($result->label === 'Medium risk') {
            $label_class = 'label-warning';
        } elseif ($result->label === 'High risk') {
            $label_class = 'label-danger';
        }

        $html = '<div class="plagiarism-detection-result">';
        $html .= '<strong>AI Detection Score:</strong> ';
        $html .= '<span class="badge ' . $label_class . '">' . round($result->score) . '%</span> ';
        $html .= '<span class="label">' . htmlspecialchars($result->label) . '</span>';
        $html .= '</div>';

        return $html;
    }

    private function is_plugin_configured() {
        $api_url = get_config('plagiarism_detection', 'detection_api_url');
        return !empty($api_url);
    }
}
