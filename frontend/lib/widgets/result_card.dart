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
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Prediction',
                  style: TextStyle(fontSize: 14, color: Colors.black54)),
                  const SizedBox(height: 4),
                  Text(
                    label,
                    style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: bg,
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(color: color.withOpacity(0.25)),
                    ),
                    child: Text(
                      'Confidence: ${confidence.toStringAsFixed(2)}',
                      style: TextStyle(color: color, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}






