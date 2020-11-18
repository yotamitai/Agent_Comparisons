from os.path import join


def merge_and_fade(output_dir, n_HLs, fade_in_frame=0, fade_out_frame=10, fade_duration=2):
    """Creates bash file to merge the HL videos and add fade in fade out effects using ffmpeg"""

    """Create the necissary files"""
    f1 = open(join(output_dir, "addFadeAndMerge.sh"), "w+")
    for i in range(n_HLs):
        f1.write(f"ffmpeg -i HL_{i}.mp4 -filter:v "
                 f"'fade=in:{fade_in_frame}:{fade_duration},fade=out:{fade_out_frame}:{fade_duration}' "
                 f"-c:v libx264 -crf 22 -preset veryfast -c:a copy fadeInOut_HL_{i}.mp4\n")
    f1.write(f"ffmpeg -f concat -safe 0 -i list.txt -c copy FUll_HL_VIDEO.mp4")
    f1.close()

    f2 = open(join(output_dir, "list.txt"), "w+")
    for j in range(n_HLs):
        f2.write(f"file fadeInOut0_HL_{j}.mp4\n")
    f2.close()

    """call ffmpeg"""
    print()
    raise Exception  # TODO this
