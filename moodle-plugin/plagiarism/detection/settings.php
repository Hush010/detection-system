<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Admin settings for AI Detection plugin
 *
 * @package    plagiarism_detection
 * @copyright  2024
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage(
        'plagiarism_detection',
        get_string('pluginname', 'plagiarism_detection')
    );

    // Enable plugin
    $settings->add(new admin_setting_configcheckbox(
        'plagiarism_detection/detection_enabled',
        get_string('enable_detection', 'plagiarism_detection'),
        get_string('enable_detection_desc', 'plagiarism_detection'),
        0
    ));

    // API URL
    $settings->add(new admin_setting_configtext(
        'plagiarism_detection/detection_api_url',
        get_string('api_url', 'plagiarism_detection'),
        get_string('api_url_desc', 'plagiarism_detection'),
        'https://detection-system-h6ee.onrender.com',
        PARAM_URL
    ));

    // Risk threshold
    $settings->add(new admin_setting_configtext(
        'plagiarism_detection/detection_threshold',
        get_string('risk_threshold', 'plagiarism_detection'),
        get_string('risk_threshold_desc', 'plagiarism_detection'),
        50,
        PARAM_INT
    ));

    $ADMIN->add('plagiarism', $settings);
}
