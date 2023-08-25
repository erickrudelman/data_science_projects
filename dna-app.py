import pandas as pd
import streamlit as st
import re
import altair as alt
from PIL import Image


# Page Title
image = Image.open('dna-logo.jpg.jpeg')
st.image(image, use_column_width=True)

st.write("""
# DNA Nucleotide Count Web App

This app counts the nucleotide composition of query DNA!

***
""")

# Input Text Box
st.header('Enter DNA sequence')

# Function to validate DNA sequence input
def is_valid_sequence(sequence):
    return re.match(r'^[ACGT]+$', sequence) is not None

# Input DNA sequence
sequence_input = st.text_area("Enter DNA sequence (ACGT only)", "").upper()

# Validate input and handle errors
if sequence_input:
    if is_valid_sequence(sequence_input):
        # DNA analysis and translation code here
        st.write("Valid DNA sequence:", sequence_input)
    else:
        st.write("Invalid DNA sequence. Please enter only A, C, G, and T.")

# DNA nucleotide count
def DNA_nucleotide_count(seq):
    d = dict([
        ('A', seq.count('A')),
        ('T', seq.count('T')),
        ('G', seq.count('G')),
        ('C', seq.count('C'))
    ])
    return d

# Perform the DNA nucleotide count
X = DNA_nucleotide_count(sequence_input)

# Print dictionary
st.subheader('1. Print dictionary')
X

# Print text
st.subheader('2. Print text')
st.write('There are ' + str(X['A']) + ' adenine (A)')
st.write('There are ' + str(X['T']) + ' thymine (T)')
st.write('There are ' + str(X['G']) + ' guanine (G)')
st.write('There are ' + str(X['C']) + ' cytosine (C)')

# Display DataFrame
st.subheader('3. Display DataFrame')
df = pd.DataFrame.from_dict(X, orient='index', columns=['count'])
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'nucleotide'})
st.write(df)

# Define a dictionary of colors for each nucleotide
color_dict = {
    'A': 'blue',
    'T': 'green',
    'G': 'orange',
    'C': 'red'
}

# Display Bar Chart using Altair
st.subheader('4. Display Bar chart')
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count',
    color=alt.Color('nucleotide:N', scale=alt.Scale(domain=list(color_dict.keys()), range=list(color_dict.values())))
)

p = p.properties(
    width=alt.Step(80)  # controls width of bar.
)

st.write(p)

# Function to calculate reverse complement
def reverse_complement(sequence):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reverse_comp_seq = ''.join(complement[base] for base in sequence[::-1])
    return reverse_comp_seq

# Streamlit app
st.title('Reverse Complement Calculator')

# Preprocess the sequence to remove whitespace characters
sequence_input = sequence_input.replace('\n', '').replace(' ', '')


# Calculate reverse complement
if sequence_input:
    reverse_comp_seq = reverse_complement(sequence_input)
    st.write("Original Sequence:", sequence_input)
    st.write("Reverse Complement:", reverse_comp_seq)


# Codon table: Mapping of codons to amino acids
codon_table = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*', 'TGA': '*',
    'TGC': 'C', 'TGT': 'C', 'TGG': 'W',
}

# Function to find open reading frames (ORFs)
def find_orfs(sequence):
    orfs = []
    start_codon = 'ATG'
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    i = 0
    while i < len(sequence) - 2:
        codon = sequence[i:i+3]
        if codon == start_codon:
            j = i + 3
            while j < len(sequence) - 2:
                codon = sequence[j:j+3]
                if codon in stop_codons:
                    orf = sequence[i:j+3]
                    orfs.append(orf)
                    i = j + 3  # Move i to next potential start codon
                    break
                j += 3
        else:
            i += 3
    
    return orfs

# Streamlit app
st.title('Codon Analysis for ORFs')

# Find ORFs and display
if sequence_input:
    orfs = find_orfs(sequence_input)
    st.write("Number of ORFs found:", len(orfs))
    st.write("Open Reading Frames:")
    for idx, orf in enumerate(orfs):
        st.write(f"ORF {idx+1}: {orf}")

st.title('DNA Translation')
st.header('DNA translation into protein sequences')
# Function to translate DNA sequence into protein sequence
def translate_dna(sequence):
    protein_sequence = ""
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        if len(codon) == 3:
            amino_acid = codon_table.get(codon, '?')
            protein_sequence += amino_acid
    return protein_sequence

# In the sequence input validation block
if sequence_input:
    if is_valid_sequence(sequence_input):
        protein_sequence = translate_dna(sequence_input)
        st.write("Valid DNA sequence:", sequence_input)
        st.write("Protein sequence:", protein_sequence)
    else:
        st.write("Invalid DNA sequence. Please enter only A, C, G, and T.")

# Save results button
if st.button('Save Results'):
     # Save analysis results to a file
        with open('results.txt', 'w') as f:
            f.write(f"DNA Nucleotide Counts: {X}\n")
            f.write(f"Protein Sequence: {protein_sequence}\n")

        # Create a download button for the saved results
        file_content = f"DNA Nucleotide Counts: {X}\nProtein Sequence: {protein_sequence}"
        st.download_button(
            label="Download Results",
            data=file_content,
            file_name="dna_analysis_results.txt",
            mime="text/plain"
        )
        st.write("Results saved and ready for download!")