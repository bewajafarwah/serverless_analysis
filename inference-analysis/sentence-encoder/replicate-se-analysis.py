import time
import replicate
import json

input = '''
Plastic pollution is increasingly affecting the health of coasts and oceans. One well-known problem is plastic bottles made from polyethylene terephthalate, or PET.
A new study involving scientists from Professor Ruth Schmitz-Streit's research group at Kiel University has shown for the first time, using microorganisms from the deep sea, that polymers such as PET are continuously degraded by an enzyme. Researchers from the University of Hamburg and the Heinrich-Heine-University Düsseldorf played a major role in the microbiological study.
The results fundamentally expand the knowledge of PET-degrading enzymes, the underlying mechanism and the evolutionary understanding of the diversity of putative PET-degrading enzymes throughout the global ocean. The research team published the results in the journal Communications Chemistry, where they discuss both biotechnological applications and the high relevance for biogeochemical processes in the ocean and on land.
The study highlights a special feature of the PET-degrading enzyme. "In our study, we have discovered a new genetic resource from deep-sea organisms belonging to the archaea," says Professor Ruth Schmitz-Streit, head of the Molecular Biology of Microorganisms working group at the Institute of General Microbiology (IfAM) and member of the research priority area Kiel Marine Science (KMS) at Kiel University. Until now, about 80 different PET-degrading enzymes were known, most of which were found in bacteria or fungi.
"Our data contribute to a better understanding of the ecological role of deep-sea archaea and the possible degradation of PET waste in the sea," says the microbiologist.
The new enzyme: PET46.Using a metagenomic approach, the research team has identified and biochemically described the PET-degrading enzyme PET46 from a non-cultured deep-sea microorganism for the first time. This involved identifying the gene from a deep-sea sample based on similarities to known sequences, synthesizing the corresponding coding gene, producing the protein in the bacterium Escherichia coli and then studying it biochemically and structurally.
PET46 has many unusual properties and adds to the scaffold diversity of PET-active enzymes. Structurally, the enzyme differs significantly from those previously discovered. For example, it has the ability to degrade both very long-chain PET molecules, known as polymers, and short-chain PET molecules, known as oligomers, which means that degradation can be continuous.Among other things, PET46 uses a completely different mechanism for substrate binding than previously known PET-degrading enzymes. The researchers describe an unusual 'lid' of 45 amino acids above the enzyme's active center as crucial for binding. In other PET enzymes, aromatic amino acids close to the active site are typical.
Promising biotechnology applications. At the molecular level, PET46 is very similar to another enzyme, ferulic acid esterase. This degrades the natural polymer lignin in plant cell walls by breaking down lignin polymers to release sugars from woody plant parts. Lignin and PET have many structural similarities, so the PET-degrading enzymes found in nature may be important for composting wood in forest soils, for example.
The biochemical properties of PET46 therefore make it a very interesting enzyme both for marine and terrestrial plastics and for biotechnology. Compared to the best-characterized PET-degrading enzymes from bacteria and composting plants, PET46 is more efficient at 70° Celsius than these reference enzymes at their respective optimum temperatures.
The research was carried out as part of the PLASTISEA project, coordinated by Professor Ute Hentschel Humeida of the GEOMAR Helmholtz Center for Ocean Research in Kiel. First author Dr. Jennifer Chow from the University of Hamburg and first author Dr. Pablo Pérez-Garcia, who works as a research assistant in Schmitz-Streit's group, contributed equally to the study. 
'''

model = replicate.models.get("MODEL_NAME")
version = model.versions.get("VERSION_NUMBER")


TRIES = 50

latency = []

for i in range(TRIES):
    prediction = replicate.predictions.create(version=version, input={"inputs":input})

    start_time = time.time()
    while prediction.status != 'succeeded':
        prediction.reload()
    elapsed_time = time.time() - start_time

    latency.append(elapsed_time)

    print(elapsed_time)


with open("replicate-se-results.json", "w") as f:
    json.dump({"replicate" : latency}, f)