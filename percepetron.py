def treinar_perceptron(dados_treino, pesos_iniciais, taxa_aprendizado, max_ciclos=10):
    # pesos_iniciais: vetor [w_0, w_1, w_2] onde w_0 é o Bias
    pesos = list(pesos_iniciais)
    alpha = taxa_aprendizado
    
    print("INICIANDO TREINAMENTO DO PERCEPTRON ")
    print(f"Pesos iniciais: [Bias (w0): {pesos[0]}, w1: {pesos[1]}, w2: {pesos[2]}]\n")
    
    for ciclo in range(1, max_ciclos + 1):
        print(f"Ciclo {ciclo} - Processando os exemplos de treinamento...")
        erros_no_ciclo = 0
        
        for i, exemplo in enumerate(dados_treino, start=1):
            # x0 é sempre 1 (Bias), x1 e x2 são os atributos normais
            x0 = 1
            x1 = exemplo['x1']
            x2 = exemplo['x2']
            y_real = exemplo['y'] # Saída real (y) do exemplo
            
            # 1. Cálculo da Soma Ponderada (u ou Σ)
            soma = (pesos[0] * x0) + (pesos[1] * x1) + (pesos[2] * x2)
            
            # 2. Aplicação da Função de Ativação (Degrau: se > 0 retorna 1, senão 0)
            if soma > 0:
                y_previsto = 1
            else:            
                y_previsto = 0

            # 3. Cálculo do Erro
            erro = y_real - y_previsto
            
            print(f"Exemplo {i} -> Entradas: ({x1}, {x2}) | Saída Real (y): {y_real}")
            print(f"  Soma Ponderada: {soma:.2f} -> Saída Prevista: {y_previsto}")
            
            # 4. Atualização de Pesos caso haja erro
            if erro != 0:
                erros_no_ciclo += 1
                pesos_antigos = list(pesos)
                
                # Fórmula do erro: wj = wj + α * erro * xj
                pesos[0] = pesos[0] + (alpha * erro * x0)
                pesos[1] = pesos[1] + (alpha * erro * x1)
                pesos[2] = pesos[2] + (alpha * erro * x2)
                
                print(f"    ERRO DETECTADO (Erro = {erro})! Atualizando pesos:")
                print(f"    Bias (w0): {pesos_antigos[0]} + ({alpha} * {erro} * {x0}) = {pesos[0]}")
                print(f"    Peso 1 (w1): {pesos_antigos[1]} + ({alpha} * {erro} * {x1}) = {pesos[1]}")
                print(f"    Peso 2 (w2): {pesos_antigos[2]} + ({alpha} * {erro} * {x2}) = {pesos[2]}")
            else:
                print(" ACERTO! Pesos mantidos.")
            print("-" * 40)
            
        # Se passar por toda a tabela sem cometer erros, o algoritmo convergiu
        if erros_no_ciclo == 0:
            print(f"\n ATENÇÃO : CONVERGÊNCIA ATINGIDA NO CICLO !!{ciclo}!")
            print(f"Pesos Finais Ajustados: [Bias (w0): {pesos[0]}, w1: {pesos[1]}, w2: {pesos[2]}]")
            return pesos
            
    print("\nO limite máximo de ciclos foi atingido antes da convergência total.")
    return pesos


# Tabela do EXERCÍCIO DA AULA 
#Aplicando função do treinamento do percpetron para o exemplo dado

# Tabela Verdade do Operador Lógico AND
dados_and = [
    {'x1': 0, 'x2': 0, 'y': 0},
    {'x1': 0, 'x2': 1, 'y': 0},
    {'x1': 1, 'x2': 0, 'y': 0},
    {'x1': 1, 'x2': 1, 'y': 1}
]

# Configurações Iniciais da tabela do exemplo
pesos_iniciais = [0.0, 0.0, 0.0]  # [w0, w1, w2]
taxa_aprendizado = 0.5            # 𝝰

# Executa o algoritmo
pesos_finais = treinar_perceptron(dados_and, pesos_iniciais, taxa_aprendizado)