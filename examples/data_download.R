
library(Seurat)
devtools::install_github("mojaveazure/seurat-disk")
devtools::install_github('satijalab/seurat-data')
library(SeuratData)
library(SeuratDisk)

options(timeout=3000)

dir.create('./data')
data_url <- 'https://xialab.s3.us-west-2.amazonaws.com/tools/PENCIL/'

# loading simulaiton_data_1
data_name <- 'PENCIL_tutorial_1'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
    download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}
load(sprintf('./data/%s.Rdata', data_name))
SaveH5Seurat(sc_data, filename = sprintf('./data/%s.h5Seurat', data_name))
Convert(sprintf('./data/%s.h5Seurat', data_name), dest='h5ad')
mvg2000 <- VariableFeatures(sc_data)
write.csv(mvg2000, sprintf('./data/%s_mvg2000_list.csv', data_name))
file.remove(sprintf('./data/%s.h5Seurat', data_name))


# loading simulaiton_data_2
data_name <- 'PENCIL_tutorial_2'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
    download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}
load(sprintf('./data/%s.Rdata', data_name))
sc_data.2@reductions$pca <- NULL
SaveH5Seurat(sc_data.2, filename = sprintf('./data/%s.h5Seurat', data_name))
Convert(sprintf('./data/%s.h5Seurat', data_name), dest='h5ad')
pcs_list <- row.names(sc_data.2)
write.csv(pcs_list, sprintf('./data/%s_pcs_list.csv', data_name))
file.remove(sprintf('./data/%s.h5Seurat', data_name))
