
library(Seurat)
devtools::install_github("mojaveazure/seurat-disk")
devtools::install_github('satijalab/seurat-data')
library(SeuratData)
library(SeuratDisk)

options(timeout=3000)

dir.create('./data')
data_url <- 'https://xialab.s3.us-west-2.amazonaws.com/tools/PENCIL/'

# loading simulaiton_data_3
data_name <- 'PENCIL_tutorial_3'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
    download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}

# loading simulaiton_data_4
data_name <- 'PENCIL_tutorial_4'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
    download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}
