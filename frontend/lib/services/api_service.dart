import 'dart:convert';
import 'dart:io' show File;

import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';


class InferenceResult {
  final String label;
  final double confidence;

  InferenceResult({required this.label, required this.confidence});
}