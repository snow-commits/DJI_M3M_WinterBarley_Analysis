import argparse

from src.m3m.radiometric import (
    check_with_nir_anchor,
    calibrate_nir_anchor,
    process_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Apply the provisional NIR-anchored M3M radiometric correction."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N capture groups (for a pilot run)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    scale, panel_reflectance, panel_signal, valid_fraction = calibrate_nir_anchor()
    print("NIR 反射率板锚定：")
    print(f"  板标称反射率：{panel_reflectance:.4f}")
    print(f"  白板有效像素比例：{valid_fraction:.2%}")
    print(f"  白板校正信号均值：{panel_signal:.8f}")
    print(f"  锚定尺度系数：{scale:.8f}")
    check_with_nir_anchor(scale)

    if args.limit is None:
        print("\n全量辐射校正开始。")
    else:
        print(f"\n开始试跑前 {args.limit} 组影像。")
    process_dataset(scale, max_groups=args.limit)


if __name__ == "__main__":
    main()
