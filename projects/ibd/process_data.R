###########################################################################################
# installing dependencies -- should be done in a clean conda env
# mamba install r-seurat=5.0.1 bioconductor-summarizedexperiment r-htmltools=0.5.7 bioconductor-singlecellexperiment

###########################################################################################
#### Build the merged and normalized Seurat objects for the CRC data

# This was taken from this paper:
# https://www.nature.com/articles/s41588-022-01088-x
# https://github.com/winstonbecker/scCRC_continuum
# https://drive.google.com/drive/folders/12j9ufV1L0uWbUlab-VoXRznDLKDO7PQ_?usp=sharing
# https://drive.google.com/drive/folders/1Kl3SSbQyYQWIzl1ZW_9AfkFJStr7sY8A?usp=sharing

# Here's the R code to generate the initial TSV:
library(Seurat)
fnames <- c(
    "raw/Final_scHTAN_colon_normal_epithelial_220213.rds",
    "raw/Final_scHTAN_colon_immune_220213.rds",
    "raw/Final_scHTAN_colon_stromal_220213.rds"
);
tmp <- lapply(fnames, FUN=function(fname){UpdateSeuratObject(readRDS(fname))})
tmp[[1]] <- AddMetaData(tmp[[1]], "colon.epithelial", col.name = "method")
tmp[[2]] <- AddMetaData(tmp[[2]], "colon.immune", col.name = "method")
tmp[[3]] <- AddMetaData(tmp[[3]], "colon.stromal", col.name = "method")

###########################################################################################
# This was taken from this paper:
# https://www.nature.com/articles/s41587-019-0332-7
# https://github.com/GreenleafLab/MPAL-Single-Cell-2019

library(Seurat)
library(SummarizedExperiment)    
library(SingleCellExperiment)
x <- as(readRDS("raw/scRNA-Healthy-Hematopoiesis-191120.rds"), "SingleCellExperiment")
x <- AddMetaData(CreateSeuratObject(counts=assays(x)$counts), x$BioClassification, col.name = "CellType")
x <- AddMetaData(x, "blood", col.name = "method")
saveRDS(x, "processed/scRNA-Healthy-Hematopoiesis-191120.seurat.rds")

###########################################################################################
# Merge everything together, and re-normalize

# merge and re-normalize
z <- merge(x, y = c(tmp[[2]], tmp[[1]], tmp[[3]]), add.cell.ids = c("", "", "", ""), project = "colon")
z <- NormalizeData(z)
z <- FindVariableFeatures(z)
z <- ScaleData(z)
z <- RunPCA(z)

# run integration
ifnb <- IntegrateLayers(object = z, method = CCAIntegration, orig.reduction = "pca", new.reduction = "integrated.cca", verbose=TRUE)
ifnb <- NormalizeData(ifnb)
ifnb <- FindVariableFeatures(ifnb)
ifnb <- ScaleData(ifnb)
ifnb <- FindNeighbors(ifnb, reduction = "integrated.cca", dims = 1:30)
ifnb <- FindClusters(ifnb, resolution = 1)
ifnb[["RNA"]] <- JoinLayers(ifnb[["RNA"]])
saveRDS(x, "processed/integrated.blood_and_colon.seurat.rds")

write.table(AggregateExpression(
    ifnb, group.by = "CellType", return.seurat = TRUE)$RNA$data, "expression/blood_and_colon.psuedobulk.expression.tsv", sep="\t") 

###########################################################################################
# Create some differential gene sets
ifnb <- readRDS("processed/integrated.blood_and_colon.seurat.rds")

markers.immune_vs_epithelial <- FindMarkers(ifnb, ident.1="colon.immune", ident.2="colon.epithelial", group.by="method")
write.table(markers.immune_vs_epithelial, "markers/markers.immune_vs_epithelial.tsv")

markers.immune_vs_stromal <- FindMarkers(ifnb, ident.1="colon.immune", ident.2="colon.stromal", group.by="method")
write.table(markers.immune_vs_stromal, "markers/markers.immune_vs_stromal.tsv")

markers.b_cell_a_vs_b_cell_b <- FindMarkers(ifnb, ident.1="17_B", ident.2=c("Memory B", "Naive B"), group.by="CellType")
write.table(markers.b_cell_a_vs_b_cell_b, "markers/markers.b_cell_a_vs_b_cell_b.tsv")





