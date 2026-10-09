def autocorregir_patente(texto):
    num_a_letra = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'}
    letra_a_num = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8', 'S': '5', 'G': '6', 'T': '7'}
    
    if len(texto) == 7:
        texto_corregido = list(texto)
        for i in [0, 1, 5, 6]:
            if texto_corregido[i] in num_a_letra: texto_corregido[i] = num_a_letra[texto_corregido[i]]
        for i in [2, 3, 4]:
            if texto_corregido[i] in letra_a_num: texto_corregido[i] = letra_a_num[texto_corregido[i]]
        return "".join(texto_corregido)
    elif len(texto) == 6:
        texto_corregido = list(texto)
        for i in [0, 1, 2]:
            if texto_corregido[i] in num_a_letra: texto_corregido[i] = num_a_letra[texto_corregido[i]]
        for i in [3, 4, 5]:
            if texto_corregido[i] in letra_a_num: texto_corregido[i] = letra_a_num[texto_corregido[i]]
        return "".join(texto_corregido)
    return texto

import collections

class GestorPatentes:
    def __init__(self, max_votos=7):
        self.historial_patentes = {}
        self.patentes_definitivas = {}
        self.max_votos = max_votos

    def obtener_patente_definitiva(self, id_auto):
        return self.patentes_definitivas.get(id_auto)

    def obtener_progreso(self, id_auto):
        if id_auto in self.historial_patentes:
            return len(self.historial_patentes[id_auto])
        return 0

    def registrar_lectura(self, id_auto, texto_detectado):
        if id_auto not in self.historial_patentes:
            self.historial_patentes[id_auto] = []
        
        self.historial_patentes[id_auto].append(texto_detectado)

        if len(self.historial_patentes[id_auto]) >= self.max_votos:
            conteo = collections.Counter(self.historial_patentes[id_auto])
            patente_ganadora = conteo.most_common(1)[0][0]
            self.patentes_definitivas[id_auto] = patente_ganadora
            return True, patente_ganadora
        
        return False, None


