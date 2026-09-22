from detection_metrics import Box, non_max_suppression, score_detections


def main() -> None:
    predictions = [
        Box(10, 10, 100, 80, 0.93, "panel"),
        Box(14, 12, 98, 78, 0.61, "panel"),
        Box(42, 30, 55, 44, 0.84, "hotspot"),
    ]
    truth = [Box(8, 9, 101, 81, label="panel"), Box(40, 29, 56, 45, label="hotspot")]
    kept = non_max_suppression(predictions)
    print("Kept detections:")
    for box in kept:
        print(" ", box)
    print("Score:", score_detections(kept, truth))


if __name__ == "__main__":
    main()
