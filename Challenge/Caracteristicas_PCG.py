import pandas as pd

df1 = pd.read_csv('Caracteristicas_Estadisticas_Grupo_a.csv',sep='\t', index_col=[0])
df2 = pd.read_csv('Caracteristicas_Estadisticas_Grupo_b.csv',sep='\t', index_col=[0])
#df3 = pd.read_csv('Desvios_Grupo_c.csv',sep='\t', index_col=[0])
df4 = pd.read_csv('Caracteristicas_Estadisticas_Grupo_d.csv',sep='\t', index_col=[0])
#df5 = pd.read_csv('Desvios_Grupo_e.csv',sep='\t', index_col=[0])
#df6 = pd.read_csv('Desvios_Grupo_f.csv',sep='\t', index_col=[0])

#vertical_concat = pd.concat([df1, df2,df3,df4,df6], axis=0)
vertical_concat = pd.concat([df1, df2,df4], axis=0)

a=vertical_concat.sort_values(by=['SNR'])

print('OK')