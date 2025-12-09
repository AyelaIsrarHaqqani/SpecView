import 'package:flutter/material.dart';

class ResultCard extends StatelessWidget {
  final String label;
  final double confidence;
  const ResultCard({super.key, required this.label, required this.confidence});
