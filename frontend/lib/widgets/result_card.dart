import 'package:flutter/material.dart';

class ResultCard extends StatelessWidget {
  final String label;
  final double confidence;
  const ResultCard({super.key, required this.label, required this.confidence});

   @override
  Widget build(BuildContext context) {
    final isHigh = confidence >= 0.8;
    final color = isHigh ? const Color(0xFF16A34A) : const Color(0xFFF59E0B);
    final bg = isHigh ? const Color(0xFFEFFBF2) : const Color(0xFFFFF7ED);
    return Card(
        child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            CircleAvatar(
              backgroundColor: color.withOpacity(0.15),
              radius: 28,
              child: Icon(Icons.shield, color: color, size: 28),
            ),



