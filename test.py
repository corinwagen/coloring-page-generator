from coloring_page import extract_boundaries

if __name__ == "__main__":
    for m in ["character", "map", "adaptive_threshold"]:
        for lt in ["thin", "normal", "thick"]:
            extract_boundaries(
                "input/france.png",
                f"output/france-{m}-{lt}.png",
                method=m,
                line_thickness=lt,
            )

            extract_boundaries(
                "input/optimus.png",
                f"output/optimus-{m}-{lt}.png",
                method=m,
                line_thickness=lt,
            )
