#!/usr/bin/env python3
"""Test script for PDF parser."""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# In Docker, agent module is directly available
from agent.ingestion.pdf_parser import parse_pdf

def test_parser(pdf_path: str):
    """Test PDF parser on a single file."""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing PDF Parser: {pdf_path.name}")
    print(f"{'='*80}\n")
    
    try:
        # Parse PDF
        udr = parse_pdf(pdf_path)
        
        # Display results
        print(f"✓ Parsing successful!")
        print(f"\n{'─'*80}")
        print(f"METADATA")
        print(f"{'─'*80}")
        print(f"Title: {udr.metadata.title}")
        print(f"Authors: {', '.join(udr.metadata.authors) if udr.metadata.authors else 'N/A'}")
        print(f"Year: {udr.metadata.publication_year or 'N/A'}")
        print(f"Pages: {udr.metadata.num_pages}")
        print(f"  - Digital: {udr.metadata.num_digital_pages}")
        print(f"  - Scanned: {udr.metadata.num_scanned_pages}")
        print(f"  - Mixed: {udr.metadata.num_mixed_pages}")
        
        if udr.metadata.abstract:
            print(f"\nAbstract: {udr.metadata.abstract[:200]}...")
        
        print(f"\n{'─'*80}")
        print(f"STRUCTURE")
        print(f"{'─'*80}")
        print(f"Pages: {len(udr.pages)}")
        print(f"Sections: {len(udr.sections)}")
        print(f"Tables: {len(udr.tables)}")
        print(f"Figures: {len(udr.figures)}")
        print(f"Equations: {len(udr.equations)}")
        print(f"References: {len(udr.references)}")
        
        total_blocks = sum(len(p.blocks) for p in udr.pages)
        total_spans = sum(len(b.spans) for p in udr.pages for b in p.blocks)
        print(f"Total Blocks: {total_blocks}")
        print(f"Total Spans: {total_spans}")
        
        print(f"\n{'─'*80}")
        print(f"EXTRACTION METHODS")
        print(f"{'─'*80}")
        for method in udr.extraction_methods_used:
            print(f"  - {method.value}")
        
        if udr.ocr_pages:
            print(f"\nOCR Applied to Pages: {udr.ocr_pages}")
        
        print(f"\n{'─'*80}")
        print(f"SAMPLE BLOCKS (First Page)")
        print(f"{'─'*80}")
        if udr.pages:
            first_page = udr.pages[0]
            print(f"Page 1: {first_page.page_type.value}, {len(first_page.blocks)} blocks\n")
            
            for i, block in enumerate(first_page.blocks[:5]):  # Show first 5 blocks
                print(f"Block {i+1} [{block.block_type}]:")
                print(f"  Text: {block.text[:100]}...")
                print(f"  Spans: {len(block.spans)}")
                if block.spans:
                    span = block.spans[0]
                    print(f"  Font: {span.font_name}, Size: {span.font_size}, Bold: {span.is_bold}")
                print()
        
        print(f"\n{'─'*80}")
        print(f"SECTIONS")
        print(f"{'─'*80}")
        for section in udr.sections[:10]:  # Show first 10 sections
            print(f"  - {section.title} (Level {section.level}, Pages {section.page_start}-{section.page_end})")
        
        if len(udr.sections) > 10:
            print(f"  ... and {len(udr.sections) - 10} more sections")
        
        if udr.tables:
            print(f"\n{'─'*80}")
            print(f"TABLES")
            print(f"{'─'*80}")
            for table in udr.tables:
                print(f"  - Page {table.page}: {len(table.data)} rows x {len(table.data[0]) if table.data else 0} cols")
                if table.caption:
                    print(f"    Caption: {table.caption}")
        
        # Serialize to JSON
        print(f"\n{'─'*80}")
        print(f"JSON SERIALIZATION")
        print(f"{'─'*80}")
        json_str = udr.model_dump_json(indent=2)
        json_size = len(json_str)
        print(f"JSON Size: {json_size:,} bytes ({json_size/1024:.2f} KB)")
        
        # Show first 500 chars of JSON
        print(f"\nJSON Preview:")
        print(json_str[:500] + "...")
        
        print(f"\n{'='*80}")
        print(f"✓ Test Complete!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_parser.py <pdf_file>")
        sys.exit(1)
    
    test_parser(sys.argv[1])
