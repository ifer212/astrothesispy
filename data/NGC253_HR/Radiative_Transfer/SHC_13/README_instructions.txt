HC3N Greenhouse models
----------------------

> ifort param_HC3N_v2.f
> ./a.out
	Hay que dar el perfil de temperatura (.tem) calculado con los modelos greenhouse y poner la luminosidad, columna de densidad de H2 y q de ese perfil y seleccionar un kabs=0


- Los modelos LTE son termalizados (i.e. LTE) y tienen todas las lineas.

- Los modelos no LTE solo incluyen hasta la v6=1 (nos faltan muchos coeficientes de Einstein para las transiciones rovibracionales.
	Transicion	vib	Jup-Jlow
	24		v=0	24->23
	26		v=0	26->25
	111		v7=1	24;1->23;-1
	117		v7=1	26;1->25;-1
	336		v6=1	24;-1->23;1
	340		v6=1	26;-1->25;1
	571		vacia
	577		vacia
	244		rovib v7=1->v=0 (26->26)
	449		rovib v6=1->v=0 (26->25)
	450		rovib v6=1->v=0 (26->27)

- Una vez hecho el modelo se puede correr con NGC253_HR_obsnLTE_modelresults_v1.py
	La funcion nLTEmodel_helper.plot_models_and_inp (que tiene variantes para hacer algunas cosas) corre los fortran que convolucionan los resultados del modelo con el beam y pintan los resultados.