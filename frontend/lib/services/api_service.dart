import 'dart:convert';
import 'dart:io' show File;

import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';


class InferenceResult {
  final String label;
  final double confidence;

  InferenceResult({required this.label, required this.confidence});
}

class ApiService {
  // Configure at runtime: flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
  static const String _baseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: 'http://127.0.0.1:8000');
