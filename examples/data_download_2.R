library(Seurat)
# devtools::install_github("mojaveazure/seurat-disk")
# devtools::install_github('satijalab/seurat-data')
# library(SeuratData)
# library(SeuratDisk)

options(timeout=3000)

dir.create('./data')
data_url <- 'https://xialab.s3.us-west-2.amazonaws.com/tools/PENCIL/'

# loading simulaiton_data_1
data_name <- 'PENCIL_tutorial_1'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
  download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}

# loading simulaiton_data_3
data_name <- 'PENCIL_tutorial_small_3'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
    download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}

# loading simulaiton_data_4
data_name <- 'PENCIL_tutorial_small_4'
if (!file.exists(sprintf('./data/%s.Rdata', data_name))){
    download.file(sprintf('%s%s.Rdata', data_url, data_name), sprintf('./data/%s.Rdata', data_name), mode='wb')
}


load('./data/PENCIL_tutorial_1.Rdata')
load('./data/PENCIL_tutorial_small_3.Rdata')
load('./data/PENCIL_tutorial_small_4.Rdata')

sc_data.raw <- sc_data
rm(sc_data)

sc_data_small@assays[["RNA"]] <- sc_data.raw@assays[["RNA"]]
sc_data_small.2@assays[["RNA"]] <- sc_data.raw@assays[["RNA"]]

sc_data <- sc_data_small
sc_data.2 <- sc_data_small.2

rm(sc_data_small)
rm(sc_data_small.2)

save(sc_data, file='./data/PENCIL_tutorial_3.Rdata')
save(sc_data.2, file='./data/PENCIL_tutorial_4.Rdata')



