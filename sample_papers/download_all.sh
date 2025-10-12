#!/bin/bash

# Array of arXiv paper URLs
urls=(
"https://arxiv.org/pdf/1706.03762.pdf"
"https://arxiv.org/pdf/1512.03385.pdf"
"https://arxiv.org/pdf/1502.03167.pdf"
"https://arxiv.org/pdf/1607.06450.pdf"
"https://arxiv.org/pdf/1810.04805.pdf"
"https://arxiv.org/pdf/2005.14165.pdf"
"https://arxiv.org/pdf/2010.11929.pdf"
"https://arxiv.org/pdf/2103.00020.pdf"
"https://arxiv.org/pdf/2002.05709.pdf"
"https://arxiv.org/pdf/1911.05722.pdf"
"https://arxiv.org/pdf/2104.14294.pdf"
"https://arxiv.org/pdf/2006.07733.pdf"
"https://arxiv.org/pdf/2006.09882.pdf"
"https://arxiv.org/pdf/1701.07875.pdf"
"https://arxiv.org/pdf/1703.10593.pdf"
"https://arxiv.org/pdf/1812.04948.pdf"
"https://arxiv.org/pdf/1912.04958.pdf"
"https://arxiv.org/pdf/2006.11239.pdf"
"https://arxiv.org/pdf/2010.02502.pdf"
"https://arxiv.org/pdf/2112.10752.pdf"
"https://arxiv.org/pdf/2208.12242.pdf"
"https://arxiv.org/pdf/2302.05543.pdf"
"https://arxiv.org/pdf/2204.06125.pdf"
"https://arxiv.org/pdf/2205.11487.pdf"
"https://arxiv.org/pdf/1910.10683.pdf"
"https://arxiv.org/pdf/1907.11692.pdf"
"https://arxiv.org/pdf/1909.11942.pdf"
"https://arxiv.org/pdf/1911.02116.pdf"
"https://arxiv.org/pdf/2205.05131.pdf"
"https://arxiv.org/pdf/2203.15556.pdf"
"https://arxiv.org/pdf/2001.08361.pdf"
"https://arxiv.org/pdf/1706.03741.pdf"
"https://arxiv.org/pdf/2203.02155.pdf"
"https://arxiv.org/pdf/2212.08073.pdf"
"https://arxiv.org/pdf/1707.06347.pdf"
"https://arxiv.org/pdf/1712.01815.pdf"
"https://arxiv.org/pdf/2304.02643.pdf"
"https://arxiv.org/pdf/2005.12872.pdf"
"https://arxiv.org/pdf/2103.14030.pdf"
"https://arxiv.org/pdf/1804.02767.pdf"
"https://arxiv.org/pdf/2004.10934.pdf"
"https://arxiv.org/pdf/1506.01497.pdf"
"https://arxiv.org/pdf/1703.06870.pdf"
"https://arxiv.org/pdf/1612.03144.pdf"
"https://arxiv.org/pdf/1505.04597.pdf"
"https://arxiv.org/pdf/1802.02611.pdf"
"https://arxiv.org/pdf/2301.12597.pdf"
"https://arxiv.org/pdf/2302.13971.pdf"
"https://arxiv.org/pdf/2307.09288.pdf"
"https://arxiv.org/pdf/2303.08774.pdf"
)

# Download each paper
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    echo "Downloading $filename..."
    curl -L "$url" -o "$filename" --silent --show-error
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded $filename"
    else
        echo "✗ Failed to download $filename"
    fi
done

echo ""
echo "All downloads complete!"
ls -lh *.pdf | wc -l | xargs echo "Total PDFs downloaded:"
