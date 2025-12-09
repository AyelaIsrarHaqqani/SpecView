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


    // Cross-platform upload: uses bytes on Web (no file.path), path on desktop/mobile.
  Future<InferenceResult> inferPlatformFile(PlatformFile file) async {
    final uri = Uri.parse('$_baseUrl/infer');
    final req = http.MultipartRequest('POST', uri);
    if (file.bytes != null) {
      req.files.add(http.MultipartFile.fromBytes('file', file.bytes!, filename: file.name));
    } else if (file.path != null) {
      req.files.add(await http.MultipartFile.fromPath('file', file.path!));
    } else {
      throw Exception('No file bytes or path available for upload.');
    }