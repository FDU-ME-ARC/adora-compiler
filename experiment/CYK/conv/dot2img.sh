#!/bin/bash

# 输出格式（可改成 png / pdf / svg）
FORMAT=png

for file in *.dot; do
    # 如果没有匹配到文件，避免报错
    [ -e "$file" ] || continue

    output="${file%.dot}.${FORMAT}"
    echo "Converting $file -> $output"

    dot -T${FORMAT} "$file" -o "$output"
done

echo "Done!"